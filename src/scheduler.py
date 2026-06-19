"""
Hierarchical Adaptive Scheduler for AetherServe.

Implements iteration-level continuous batching with a three-tier cascading
eviction policy:

  Tier 1 (GPU) OOM  → try swap_gpu_to_cpu
  Tier 2 (CPU) also full → find coldest CPU-swapped request by last_accessed_time,
                           cascade it to Tier 3 storage, then retry GPU→CPU
  Both full             → recompute (drop blocks, re-queue as WAITING)

The elevator logic runs every step in reverse:
  Tier 3 → Tier 2 when CPU has room (passive restore)
  Tier 2 → Tier 1 when GPU batch is not full (admit to running)

Predictive prefetch hook: the PredictivePrefetcher calls tick() once per
engine step to proactively move soon-needed storage blocks up to CPU before
the scheduler formally requests them, hiding Tier 3 I/O latency.

Scheduling priority within queues: FCFS (arrival order).
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from src.cache_manager import HierarchicalCacheManager
from src.models import SequenceRequest, SequenceStatus


@dataclass
class SchedulerOutputs:
    """
    Snapshot of what happened during one schedule() call.

    scheduled       — sequences that will execute in this engine step
    preempted       — sequences evicted from GPU (finished OR OOM)
    swapped_cpu_out — sequences moved Tier 1 → Tier 2 this step
    swapped_cpu_in  — sequences restored Tier 2 → Tier 1 this step
    swapped_storage_out — sequences moved Tier 2 → Tier 3 this step
    swapped_storage_in  — sequences restored Tier 3 → Tier 2 this step
    num_batched_tokens  — total token budget consumed this step
                          (BLOCK_SIZE * num_prefill_tokens + 1 per decode)

    num_batched_tokens example:
      User A (new, 40-token prompt) + User B (new, 10-token prompt)
      + User C (running, generating) + User D (running, generating)
      → num_batched_tokens = 40 + 10 + 1 + 1 = 52
    """
    scheduled: List[SequenceRequest] = field(default_factory=list)
    preempted: List[SequenceRequest] = field(default_factory=list)
    swapped_cpu_out: List[SequenceRequest] = field(default_factory=list)
    swapped_cpu_in: List[SequenceRequest] = field(default_factory=list)
    swapped_storage_out: List[SequenceRequest] = field(default_factory=list)
    swapped_storage_in: List[SequenceRequest] = field(default_factory=list)
    num_batched_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class HierarchicalAdaptiveScheduler:
    """
    Three-tier continuous batching scheduler.

    Queues:
      waiting_queue         — requests not yet allocated any GPU memory
      running_queue         — requests actively executing on GPU
      swapped_cpu_queue     — requests with blocks in CPU DRAM (Tier 2)
      swapped_storage_queue — requests with blocks in NVMe storage (Tier 3)
    """

    def __init__(
        self,
        cache_manager: HierarchicalCacheManager,
        max_batch_size: int = 8,
    ) -> None:
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size

        self.waiting_queue: deque[SequenceRequest] = deque()
        self.running_queue: List[SequenceRequest] = []
        self.swapped_cpu_queue: deque[SequenceRequest] = deque()
        self.swapped_storage_queue: deque[SequenceRequest] = deque()

    def add_request(self, request: SequenceRequest) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting_queue.append(request)

    def schedule(self) -> SchedulerOutputs:
        outputs = SchedulerOutputs()

        # Phase 1: Extend running sequences
        # For each running request, try to secure a GPU slot for the next
        # token. If the GPU is full, cascade the request down a tier.
        still_running: List[SequenceRequest] = []

        for req in self.running_queue:
            req.last_accessed_time = time.monotonic()

            if req.is_finished:
                req.status = SequenceStatus.FINISHED
                self.cache_manager.free(req.request_id)
                outputs.preempted.append(req)
                continue

            if self.cache_manager.append_slot(req):
                # Slot secured — this request runs in this step
                still_running.append(req)
                outputs.scheduled.append(req)
                outputs.num_batched_tokens += 1
            else:
                # GPU OOM: begin cascading eviction
                self._cascade_evict(req, outputs)

        self.running_queue = still_running

        # Phase 2: Restore elevator (Tier 3 → Tier 2 → Tier 1)
        # Tier 3 → Tier 2: restore storage-swapped requests to CPU if room
        still_storage: deque[SequenceRequest] = deque()
        while self.swapped_storage_queue:
            req = self.swapped_storage_queue.popleft()
            if self.cache_manager.swap_storage_to_cpu(req):
                req.status = SequenceStatus.SWAPPED_CPU
                self.swapped_cpu_queue.appendleft(req)
                outputs.swapped_storage_in.append(req)
            else:
                still_storage.append(req)
        self.swapped_storage_queue = still_storage

        # Tier 2 → Tier 1: admit CPU-swapped requests to GPU if batch has room
        still_cpu: deque[SequenceRequest] = deque()
        while self.swapped_cpu_queue:
            if len(self.running_queue) >= self.max_batch_size:
                still_cpu.extend(self.swapped_cpu_queue)
                break
            req = self.swapped_cpu_queue.popleft()
            if self.cache_manager.swap_cpu_to_gpu(req):
                req.status = SequenceStatus.RUNNING
                self.running_queue.append(req)
                outputs.swapped_cpu_in.append(req)
                outputs.scheduled.append(req)
                outputs.num_batched_tokens += 1
            else:
                still_cpu.append(req)
        if still_cpu:
            self.swapped_cpu_queue.extendleft(reversed(list(still_cpu)))

        # Phase 3: Admit new requests (prefill)
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size:
            candidate = self.waiting_queue[0]
            if not self.cache_manager.can_allocate(candidate):
                break  # can't allocate, hold the line (FIFO, no starvation)
            if self.cache_manager.allocate(candidate):
                candidate.status = SequenceStatus.RUNNING
                self.waiting_queue.popleft()
                self.running_queue.append(candidate)
                outputs.scheduled.append(candidate)
                outputs.num_batched_tokens += len(candidate.prompt_tokens)
            else:
                break

        return outputs

    def _cascade_evict(
        self, req: SequenceRequest, outputs: SchedulerOutputs
    ) -> None:
        """
        GPU OOM eviction cascade for a single sequence.

        Step 1: try GPU → CPU.
        Step 2: if CPU full, find the coldest CPU-swapped request
                (min last_accessed_time) and push it to Tier 3 storage
                to free a CPU slot, then retry GPU → CPU.
        Step 3: if storage is also full, drop the request entirely and
                re-queue for full recomputation.
        """
        if self.cache_manager.swap_gpu_to_cpu(req):
            req.status = SequenceStatus.SWAPPED_CPU
            self.swapped_cpu_queue.appendleft(req)
            outputs.swapped_cpu_out.append(req)
            outputs.preempted.append(req)
            return

        # CPU also full — cascade the coldest CPU resident to Tier 3
        cold = self._coldest_cpu_request()
        if cold is not None and self.cache_manager.swap_cpu_to_storage(cold):
            cold.status = SequenceStatus.SWAPPED_STORAGE
            try:
                self.swapped_cpu_queue.remove(cold)
            except ValueError:
                pass
            self.swapped_storage_queue.append(cold)
            outputs.swapped_storage_out.append(cold)

            # Retry now that a CPU slot opened up
            if self.cache_manager.swap_gpu_to_cpu(req):
                req.status = SequenceStatus.SWAPPED_CPU
                self.swapped_cpu_queue.appendleft(req)
                outputs.swapped_cpu_out.append(req)
                outputs.preempted.append(req)
                return

        # All tiers exhausted — recompute (wipe blocks, re-queue as WAITING)
        req.status = SequenceStatus.WAITING
        self.cache_manager.free(req.request_id)
        self.waiting_queue.appendleft(req)
        outputs.preempted.append(req)

    def _coldest_cpu_request(self) -> Optional[SequenceRequest]:
        """Return the CPU-swapped request with the oldest last_accessed_time."""
        if not self.swapped_cpu_queue:
            return None
        return min(self.swapped_cpu_queue, key=lambda r: r.last_accessed_time)

    def has_unfinished_requests(self) -> bool:
        return bool(
            self.waiting_queue
            or self.running_queue
            or self.swapped_cpu_queue
            or self.swapped_storage_queue
        )

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_queue),
            "swapped_cpu": len(self.swapped_cpu_queue),
            "swapped_storage": len(self.swapped_storage_queue),
        }
