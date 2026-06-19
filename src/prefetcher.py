"""
Predictive KV-cache prefetcher for AetherServe.

Inspired by KVSwap (ArXiv 2024): proactively migrate Tier 3 → Tier 2 for
requests that are likely to be scheduled soon, hiding NVMe I/O latency
behind ongoing GPU decode work.

The key insight: when running sequences are close to finishing (within
`lookahead` tokens), GPU slots will open up shortly. Starting the Tier 3
→ Tier 2 transfer now means the data is in CPU RAM by the time the
scheduler tries to promote it to GPU — zero stall.

tick() is called synchronously once per engine step, before schedule().
It checks how many GPU slots are about to free up and pre-fetches that
many requests from the storage queue into CPU DRAM.

Because asyncio is single-threaded, there is no data race: tick() and
schedule() never run concurrently, so queue mutations are safe.
"""

import time
from typing import TYPE_CHECKING

from src.cache_manager import HierarchicalCacheManager
from src.models import SequenceStatus

if TYPE_CHECKING:
    from src.scheduler import HierarchicalAdaptiveScheduler


class PredictivePrefetcher:
    """
    Pre-fetches Tier 3 blocks into Tier 2 before the scheduler needs them.

    Algorithm:
      1. Count running requests within `lookahead` tokens of finishing.
         These are "almost-done" — they will free GPU slots very soon.
      2. For each about-to-free slot, move the front request from the
         storage queue to the CPU queue now.
      3. The scheduler's Phase 2 will then find these in swapped_cpu_queue
         and elevate them to GPU without hitting the storage I/O delay.
    """

    def __init__(
        self,
        cache_manager: HierarchicalCacheManager,
        scheduler: "HierarchicalAdaptiveScheduler",
        lookahead: int = 3,
    ) -> None:
        self._cache = cache_manager
        self._scheduler = scheduler
        self.lookahead = lookahead
        self.total_prefetches: int = 0

    def tick(self) -> int:
        """
        Run one prefetch cycle. Returns the number of blocks moved Tier3→Tier2.
        Called once per engine step, before schedule().
        """
        if not self._scheduler.swapped_storage_queue:
            return 0

        # How many running sequences will finish within `lookahead` steps?
        slots_opening = sum(
            1
            for r in self._scheduler.running_queue
            if r.max_output_tokens - len(r.generated_tokens) <= self.lookahead
        )
        if slots_opening == 0:
            return 0

        moved = 0
        candidates = list(self._scheduler.swapped_storage_queue)[:slots_opening]
        for req in candidates:
            if req.status != SequenceStatus.SWAPPED_STORAGE:
                continue
            if self._cache.swap_storage_to_cpu(req):
                req.status = SequenceStatus.SWAPPED_CPU
                try:
                    self._scheduler.swapped_storage_queue.remove(req)
                except ValueError:
                    pass
                self._scheduler.swapped_cpu_queue.appendleft(req)
                moved += 1

        self.total_prefetches += moved
        return moved
