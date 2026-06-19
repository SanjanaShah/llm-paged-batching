"""
Core data structures for AetherServe.

SequenceStatus now distinguishes three inactive states:
  SWAPPED_CPU     — KV-cache blocks moved to Tier 2 (host DRAM)
  SWAPPED_STORAGE — KV-cache blocks moved to Tier 3 (NVMe / storage file)

last_accessed_time drives the LRU eviction policy: when the GPU hits OOM,
the scheduler selects the coldest (smallest timestamp) CPU-tier request to
cascade down to Tier 3 first.

prompt_text carries the raw string for real-model tokenization. If only
prompt_tokens is supplied (as in tests), the engine skips tokenization.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

BLOCK_SIZE = 16  # KV-pairs per physical memory block (matches vLLM default)

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    SWAPPED_CPU = auto()      # Tier 2: host RAM
    SWAPPED_STORAGE = auto()  # Tier 3: NVMe / storage file
    FINISHED = auto()
    ABORTED = auto()

@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0

@dataclass
class SequenceRequest:
    request_id: str
    prompt_tokens: List[int]
    max_output_tokens: int
    prompt_text: Optional[str] = None
    generated_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = field(default=SequenceStatus.WAITING)
    arrival_time: float = field(default_factory=time.monotonic)
    last_accessed_time: float = field(default_factory=time.monotonic)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def total_length(self) -> int:
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def is_finished(self) -> bool:
        return len(self.generated_tokens) >= self.max_output_tokens

    def num_logical_blocks_needed(self) -> int:
        return max(1, (self.total_length + BLOCK_SIZE - 1) // BLOCK_SIZE)

    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def total_latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time
