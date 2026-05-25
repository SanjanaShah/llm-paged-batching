from dataclasses import dataclass, field
from typing import List
from src.models import SequenceRequest, SequenceStatus
from src.cache_manager import CacheManager


@dataclass
class SchedulerOutputs:
    """
    Describes the exact set of sequences to run in the next engine step.

    scheduled: sequences that will participate in this forward pass iteration
    preempted: sequences evicted this step (either finished or OOM)
    swapped_in: sequences restored from CPU to GPU this step
    swapped_out: sequences evicted from GPU to CPU this step
    num_batched_tokens: total tokens sent through the model (prefill + 1 per decode)
    """
    scheduled: List[SequenceRequest] = field(default_factory=list)
    preempted: List[SequenceRequest] = field(default_factory=list)
    swapped_in: List[SequenceRequest] = field(default_factory=list)
    swapped_out: List[SequenceRequest] = field(default_factory=list)
    num_batched_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class AetherserveScheduler:
    """
    Iteration-level continuous batching scheduler.

    Unlike static batching (wait for all N sequences to finish), this
    scheduler re-evaluates the batch composition at every single decode step.
    Finished sequences are immediately evicted and their physical memory freed,
    allowing new requests from the waiting queue to be injected mid-batch.

    Scheduling policy:
    1. Run phase: extend all currently-running sequences by one token slot.
       Preempt any that hit OOM (swap-first, then recompute fallback).
    2. Swap-in phase: restore previously swapped sequences if GPU has room.
    3. Prefill phase: admit new waiting requests up to max_batch_size.

    Priority within queues: FCFS (arrival order). No starvation prevention
    in this implementation — a production system would add aging or
    per-tenant priority weights.
    """

    def __init__(self, cache_manager: CacheManager, max_batch_size: int = 8):
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size
        self.waiting_queue: List[SequenceRequest] = []
        self.running_queue: List[SequenceRequest] = []
        self.swapped_queue: List[SequenceRequest] = []

    def add_request(self, request: SequenceRequest) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting_queue.append(request)

    def schedule(self) -> SchedulerOutputs:
        outputs = SchedulerOutputs()

        # --- Phase 1: Extend running sequences ---
        still_running: List[SequenceRequest] = []
        for req in self.running_queue:
            if req.is_finished:
                req.status = SequenceStatus.FINISHED
                self.cache_manager.free(req.request_id)
                outputs.preempted.append(req)
                continue

            if self.cache_manager.append_slot(req):
                still_running.append(req)
                outputs.scheduled.append(req)
                outputs.num_batched_tokens += 1
            else:
                # OOM: try swap-out to CPU, fall back to recompute (re-queue WAITING)
                if self.cache_manager.swap_out(req):
                    req.status = SequenceStatus.SWAPPED
                    self.swapped_queue.insert(0, req)
                    outputs.swapped_out.append(req)
                else:
                    req.status = SequenceStatus.WAITING
                    self.cache_manager.free(req.request_id)
                    self.waiting_queue.insert(0, req)
                outputs.preempted.append(req)

        self.running_queue = still_running

        # --- Phase 2: Swap in preempted sequences ---
        for req in list(self.swapped_queue):
            if len(self.running_queue) >= self.max_batch_size:
                break
            if self.cache_manager.swap_in(req):
                req.status = SequenceStatus.RUNNING
                self.swapped_queue.remove(req)
                self.running_queue.append(req)
                outputs.swapped_in.append(req)
                outputs.scheduled.append(req)
                outputs.num_batched_tokens += 1

        # --- Phase 3: Admit new requests (prefill) ---
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size:
            candidate = self.waiting_queue[0]
            if not self.cache_manager.can_allocate(candidate):
                break  # Memory pressure — hold the line, don't admit
            if self.cache_manager.allocate(candidate):
                candidate.status = SequenceStatus.RUNNING
                self.waiting_queue.pop(0)
                self.running_queue.append(candidate)
                outputs.scheduled.append(candidate)
                outputs.num_batched_tokens += len(candidate.prompt_tokens)
            else:
                break

        return outputs

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting_queue or self.running_queue or self.swapped_queue)

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_queue),
            "swapped": len(self.swapped_queue),
        }
