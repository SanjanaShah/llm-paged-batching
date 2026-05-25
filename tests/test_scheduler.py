import pytest
from src.models import SequenceRequest, SequenceStatus, BLOCK_SIZE
from src.cache_manager import CacheManager
from src.scheduler import AetherserveScheduler


def _req(rid: str, prompt_len: int = BLOCK_SIZE, max_output: int = 4) -> SequenceRequest:
    return SequenceRequest(
        request_id=rid,
        prompt_tokens=list(range(prompt_len)),
        max_output_tokens=max_output,
    )


def _make_scheduler(gpu_blocks: int = 64, cpu_blocks: int = 64, batch: int = 4) -> AetherserveScheduler:
    return AetherserveScheduler(
        cache_manager=CacheManager(num_gpu_blocks=gpu_blocks, num_cpu_blocks=cpu_blocks),
        max_batch_size=batch,
    )


def test_admits_waiting_requests():
    sched = _make_scheduler()
    reqs = [_req(f"r{i}") for i in range(3)]
    for r in reqs:
        sched.add_request(r)

    out = sched.schedule()
    assert len(out.scheduled) == 3
    assert all(r.status == SequenceStatus.RUNNING for r in reqs)


def test_respects_max_batch_size():
    sched = _make_scheduler(batch=2)
    for i in range(5):
        sched.add_request(_req(f"r{i}"))

    out = sched.schedule()
    assert len(sched.running_queue) <= 2
    assert len(sched.waiting_queue) >= 3


def test_finished_sequences_evicted_next_step():
    sched = _make_scheduler()
    req = _req("r0", max_output=1)
    sched.add_request(req)

    # Admit
    sched.schedule()
    assert req.status == SequenceStatus.RUNNING

    # Generate the one token it needs
    req.generated_tokens.append(42)
    assert req.is_finished

    out = sched.schedule()
    assert req not in sched.running_queue
    assert req.status == SequenceStatus.FINISHED


def test_continuous_injection_after_finish():
    """A short sequence finishing mid-batch should open a slot for a waiting request."""
    sched = _make_scheduler(gpu_blocks=64, batch=2)

    short = _req("short", max_output=1)
    long_ = _req("long", max_output=20)
    waiting = _req("waiter", max_output=5)

    sched.add_request(short)
    sched.add_request(long_)
    sched.add_request(waiting)

    # Step 1: admit short + long (batch full)
    sched.schedule()
    assert waiting.status == SequenceStatus.WAITING

    # Finish short
    short.generated_tokens.append(42)
    assert short.is_finished

    # Step 2: evict short, inject waiter
    sched.schedule()
    assert waiting.status == SequenceStatus.RUNNING


def test_preemption_on_oom():
    """When GPU is full and a running sequence needs a new block, it must be preempted."""
    cache = CacheManager(num_gpu_blocks=2, num_cpu_blocks=16)
    sched = AetherserveScheduler(cache_manager=cache, max_batch_size=8)

    r1 = _req("r1", prompt_len=BLOCK_SIZE, max_output=BLOCK_SIZE + 2)
    r2 = _req("r2", prompt_len=BLOCK_SIZE, max_output=BLOCK_SIZE + 2)
    sched.add_request(r1)
    sched.add_request(r2)

    # Both admitted, consuming all 2 GPU blocks
    sched.schedule()
    assert len(sched.running_queue) == 2

    # Generate tokens until both sequences need a 2nd block
    for req in [r1, r2]:
        for _ in range(BLOCK_SIZE):
            req.generated_tokens.append(42)

    # Next schedule: 0 free GPU blocks, both need growth → at least one preempted
    out = sched.schedule()
    total_preempted_or_swapped = len(out.swapped_out) + sum(
        1 for r in out.preempted if r.status in (SequenceStatus.WAITING, SequenceStatus.SWAPPED)
    )
    assert total_preempted_or_swapped >= 1


def test_swap_in_after_gpu_pressure_drops():
    """A swapped sequence should be restored once GPU memory becomes available."""
    cache = CacheManager(num_gpu_blocks=2, num_cpu_blocks=4)
    sched = AetherserveScheduler(cache_manager=cache, max_batch_size=8)

    r1 = _req("r1", prompt_len=BLOCK_SIZE, max_output=BLOCK_SIZE + 2)
    r2 = _req("r2", prompt_len=BLOCK_SIZE, max_output=1)
    sched.add_request(r1)
    sched.add_request(r2)

    sched.schedule()  # Both admitted, 2 blocks used

    # Push r1 past its first block boundary to trigger OOM on next schedule
    for _ in range(BLOCK_SIZE):
        r1.generated_tokens.append(42)

    # r2 finishes
    r2.generated_tokens.append(42)

    # Schedule: r2 evicted (finished), r1 may be preempted since 0 free blocks for growth
    out = sched.schedule()

    # If r1 was swapped, next schedule with freed r2 blocks should swap it back in
    if r1.status == SequenceStatus.SWAPPED:
        out2 = sched.schedule()
        assert r1.status == SequenceStatus.RUNNING


def test_has_unfinished_requests():
    sched = _make_scheduler()
    assert not sched.has_unfinished_requests()

    sched.add_request(_req("r0"))
    assert sched.has_unfinished_requests()
