"""
Core Data Models

This file defines data structures and enums for SequenceRequests and PhysicalBlocks.
It contains functions to calculate block allocations to simulate a zero-fragmentation virtual memory layout.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

BLOCK_SIZE = 16  # KV-pairs per physical memory block (matches vLLM default)

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    SWAPPED = auto()
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
    generated_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = field(default=SequenceStatus.WAITING)
    arrival_time: float = field(default_factory=time.monotonic)
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
