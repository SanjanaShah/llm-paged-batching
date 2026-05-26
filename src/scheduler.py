from dataclasses import dataclass, field
from typing import List
from src.models import SequenceRequest, SequenceStatus
from src.cache_manager import CacheManager
from collections import deque


@dataclass
class SchedulerOutputs:
    """
    Describes the exact set of sequences to run in the next engine step.

    scheduled: sequences that will participate in this forward pass iteration
    preempted: sequences evicted this step (either finished or OOM)
    swapped_in: sequences restored from CPU to GPU this step
    swapped_out: sequences evicted from GPU to CPU this step
    num_batched_tokens: total tokens sent through the model (prefill + 1 per decode)

    num_batched_tokens math:
    User 1 (new): prompt 40 tokens, User 2 (new): prompt 10 tokens
    User 3 (running): history 500 tokens + generating, User 4 (running): history 12 tokens + generating
    num_batched_tokens = 40+10+1+1 = 52 tokens
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
    Continuous batching scheduler.

    Each scheduling step dynamically decides which requests run,
    wait, swap to CPU memory, or finish. This allows the engine
    to efficiently share GPU memory across many concurrent users.
    """

    def __init__(self, cache_manager: CacheManager, max_batch_size: int = 8):
        self.cache_manager = cache_manager
        self.max_batch_size = max_batch_size
        
        # deques for efficient front/back modifications
        self.waiting_queue: deque[SequenceRequest] = deque()
        self.swapped_queue: deque[SequenceRequest] = deque()

        self.running_queue: List[SequenceRequest] = []

    def add_request(self, request: SequenceRequest) -> None:
        request.status = SequenceStatus.WAITING
        self.waiting_queue.append(request) # req comes in, append to waiting
        # req contains id, prompt tokens, max out tokens, tokens generated
        #     status (default waiting), time metrics

    def schedule(self) -> SchedulerOutputs:
        outputs = SchedulerOutputs() # create scheduler output for results of iteration

        # << Address currently RUNNING tasks >>
        still_running: List[SequenceRequest] = [] # if not done
        for req in self.running_queue:
            if req.is_finished:
                req.status = SequenceStatus.FINISHED
                self.cache_manager.free(req.request_id) # free if finished
                outputs.preempted.append(req) # add to done
                continue

            # req is not finished
            if self.cache_manager.append_slot(req): # check if another slot is needed/get, proceed
                still_running.append(req)
                outputs.scheduled.append(req)
                outputs.num_batched_tokens += 1
            else: # can't give another slot, OOM: try swap-out to CPU, fall back to WAITING
                if self.cache_manager.swap_out(req): # if swap to CPU, good
                    req.status = SequenceStatus.SWAPPED
                    self.swapped_queue.appendleft(req) # appendleft to handle it sooner
                    outputs.swapped_out.append(req)
                else:
                    # Treat as a new request for recomputation
                    # CPU - raw text of prompt + partial response is not wiped
                    # GPU - wipe all dynamically allocated blocks holding context math (prompt + response)
                    req.status = SequenceStatus.WAITING
                    self.cache_manager.free(req.request_id)
                    self.waiting_queue.appendleft(req)
                outputs.preempted.append(req)

        self.running_queue = still_running # set for next iteration

        # << Bring back swapped tasks (these were paused, but mem is in CPU) >>
        still_swapped: deque[SequenceRequest] = deque() # these will remain swapped
        
        while self.swapped_queue:
            # Check the batch limit
            # If the GPU batch is already full, we can stop evaluating immediately
            if len(self.running_queue) >= self.max_batch_size:
                break 

            req = self.swapped_queue.popleft()
                
            if self.cache_manager.swap_in(req):
                req.status = SequenceStatus.RUNNING
                self.running_queue.append(req)
                outputs.swapped_in.append(req)
                outputs.scheduled.append(req)
                outputs.num_batched_tokens += 1
            else:
                still_swapped.append(req)  # Put back, GPU lacks mem
                
        # If we broke early because the batch was full, add any 
        # un-evaluated requests back
        if still_swapped:
            self.swapped_queue.extendleft(reversed(still_swapped))

        # ex. scenario: req A (large), req B (small), req C (medium)
        # gpu doesnt have mem for req A -> still_swapped
        # req B swapped in -> running_queue, batch at max cap, STOP!
        # reverse: req A, X, Z in still_swapped, reverse to Z, X, A, push to front: A, X, Z, ...
        # swapped_queue looks like: A, C
        self.swapped_queue.extendleft(reversed(still_swapped)) if still_swapped else None

        # << Pre-fill (new reqs in waiting_queue) >>
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size: # have req space we can process
            candidate = self.waiting_queue[0] # peek
            if not self.cache_manager.can_allocate(candidate):
                break
            # we do not loop through rest, because otherwise front user will be put on hold forever
            # fifo - first in has priority

            if self.cache_manager.allocate(candidate):
                candidate.status = SequenceStatus.RUNNING
                self.waiting_queue.popleft()
                self.running_queue.append(candidate)
                outputs.scheduled.append(candidate)
                outputs.num_batched_tokens += len(candidate.prompt_tokens)
            else:
                break

        return outputs

    # returns true if there is a req in waiting or decoding or paused
    # we can shut down if there are no active reqs
    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting_queue or self.running_queue or self.swapped_queue)

    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_queue),
            "swapped": len(self.swapped_queue),
        }