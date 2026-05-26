from typing import Dict, List, Optional
from src.models import PhysicalBlock, SequenceRequest


class BlockAllocator:
    """
    Pool allocator for a fixed set of physical memory blocks.

    Models a GPU HBM allocator: fixed total capacity, O(1) alloc/free.
    Using a list as a free-stack gives cache-local block IDs on reuse.
    """

    def __init__(self, num_blocks: int):
        self.num_total_blocks = num_blocks
        self._free: List[PhysicalBlock] = [PhysicalBlock(block_id=i) for i in range(num_blocks)]
        self._allocated: Dict[int, PhysicalBlock] = {}

    def allocate(self) -> Optional[PhysicalBlock]:
        if not self._free:
            return None
        block = self._free.pop()
        block.ref_count = 1
        self._allocated[block.block_id] = block
        return block

    def free(self, block: PhysicalBlock) -> None:
        block.ref_count -= 1
        if block.ref_count == 0:
            del self._allocated[block.block_id]
            self._free.append(block)

    def num_free_blocks(self) -> int:
        return len(self._free)

    def num_allocated_blocks(self) -> int:
        return len(self._allocated)


class CacheManager:
    """
    Virtual KV-cache memory manager implementing PagedAttention.

    Each sequence's KV-cache is split into fixed-size logical blocks that map
    to non-contiguous physical blocks on GPU HBM. This eliminates external
    fragmentation entirely: freed blocks are immediately reusable by any
    subsequent request regardless of its size or shape.

    block_table[request_id] = [phys_block_0, phys_block_1, ...]
    where logical block i maps to phys_block_i.

    CPU blocks back a swap tier: sequences preempted from GPU are serialized
    to CPU memory and can be swapped back in when GPU pressure drops.
    """

    def __init__(self, num_gpu_blocks: int, num_cpu_blocks: int = 0):
        self._gpu = BlockAllocator(num_gpu_blocks)
        self._cpu = BlockAllocator(num_cpu_blocks)
        self._gpu_table: Dict[str, List[PhysicalBlock]] = {}
        self._cpu_table: Dict[str, List[PhysicalBlock]] = {}

    def can_allocate(self, request: SequenceRequest) -> bool:
        return self._gpu.num_free_blocks() >= request.num_logical_blocks_needed()

    def allocate(self, request: SequenceRequest) -> bool:
        needed = request.num_logical_blocks_needed()
        if self._gpu.num_free_blocks() < needed:
            return False
        
        gpu_blocks = []
        for _ in range(needed):
            block = self._gpu.allocate()
            if block is None: # allocation fails
                return False
            gpu_blocks.append(block)
        self._gpu_table[request.request_id] = gpu_blocks
        return True

    def append_slot(self, request: SequenceRequest) -> bool:
        """
        Lazily grows a sequence's block table by one physical block only when
        a new block boundary is crossed. Called once per decode step per active sequence.
        Returns False if GPU is OOM - caller must preempt this sequence (paused till
        active memory).
        """
        needed = request.num_logical_blocks_needed()
        current = self._gpu_table.get(request.request_id, [])

        # if we need to allocate 1 more block
        if needed > len(current):
            block = self._gpu.allocate()
            if block is None: # allocation fails
                return False
            current.append(block)
            self._gpu_table[request.request_id] = current
        return True

    def free(self, request_id: str) -> None:
        for block in self._gpu_table.pop(request_id, []):
            self._gpu.free(block)

    def swap_out(self, request: SequenceRequest) -> bool:
        """
        Evict a sequence from GPU to CPU (swap-out preemption).
        GPU blocks are freed immediately and moved to CPU.
        Returns False if CPU is also full — caller falls back to recompute.
        """
        gpu_blocks = self._gpu_table.pop(request.request_id, [])
        if not gpu_blocks:
            return False
        if self._cpu.num_free_blocks() < len(gpu_blocks):
            for b in gpu_blocks:
                # kill the req, there are no free blocks in GPU and CPU to handle this req
                self._gpu.free(b)
            return False

        cpu_blocks = []
        for _ in gpu_blocks:
            block = self._cpu.allocate()
            if block is None: # allocation fails
                for b in cpu_blocks: # clear cpu blocks so far
                    self._cpu.free(b)
                return False
            cpu_blocks.append(block)
        self._cpu_table[request.request_id] = cpu_blocks

        # free gpu
        for b in gpu_blocks:
            self._gpu.free(b)
        return True

    def swap_in(self, request: SequenceRequest) -> bool:
        """
        Restore a swapped sequence from CPU back to GPU (swap-in).
        Returns False if GPU still lacks space — sequence stays on CPU.
        """
        cpu_blocks = self._cpu_table.get(request.request_id, [])
        if not cpu_blocks: # if not in cpu
            return False
        if self._gpu.num_free_blocks() < len(cpu_blocks): # no space in gpu
            return False

        new_gpu_blocks = []
        for _ in cpu_blocks:
            block = self._gpu.allocate()
            if block is None: # allocation in gpu fails
                for b in new_gpu_blocks: # clear gpu blocks so far
                    self._gpu.free(b)
                return False
            new_gpu_blocks.append(block)
        self._gpu_table[request.request_id] = new_gpu_blocks

        for b in cpu_blocks:
            self._cpu.free(b)
        del self._cpu_table[request.request_id]
        return True

    # Example Output:
    # >>> cache_manager.block_table_for("req-042")
    # [4, 82, 19]
    def block_table_for(self, request_id: str) -> List[int]:
        return [b.block_id for b in self._gpu_table.get(request_id, [])]

    def memory_stats(self) -> dict:
        gpu_total = self._gpu.num_total_blocks
        gpu_used = self._gpu.num_allocated_blocks()
        return {
            "gpu_total_blocks": gpu_total,
            "gpu_used_blocks": gpu_used,
            "gpu_free_blocks": self._gpu.num_free_blocks(),
            "gpu_utilization": gpu_used / gpu_total if gpu_total > 0 else 0.0,
            "cpu_total_blocks": self._cpu.num_total_blocks,
            "cpu_used_blocks": self._cpu.num_allocated_blocks(),
            "cpu_free_blocks": self._cpu.num_free_blocks(),
            "num_active_sequences": len(self._gpu_table),
            "num_swapped_sequences": len(self._cpu_table),
        }
