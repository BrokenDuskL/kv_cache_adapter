from __future__ import annotations

import torch

if __package__ in (None, ""):
    from adapter import InMemoryBlockStoreBackend, KVCacheAdapter
else:
    from .adapter import InMemoryBlockStoreBackend, KVCacheAdapter


def main() -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "npu") and torch.npu.is_available():
        device = torch.device("npu")
    else:
        device = torch.device("cpu")

    backend = InMemoryBlockStoreBackend(
        {
            0: torch.tensor([0.0, 10.0], device=device),
            1: torch.tensor([1.0, 11.0], device=device),
            2: torch.tensor([2.0, 12.0], device=device),
            3: torch.tensor([3.0, 13.0], device=device),
        }
    )
    adapter = KVCacheAdapter(
        num_actual_blocks=2,
        num_logical_blocks=4,
        actual_blocks=torch.zeros((2, 2), dtype=torch.float32, device=device),
        backend=backend,
    )
    print("runtime path ->", adapter.runtime_path)

    first_mapping = adapter.load(torch.tensor([0, 1], dtype=torch.int64, device=device))
    print("load [0, 1] ->", first_mapping)

    adapter.release(torch.tensor([0], dtype=torch.int64, device=device))
    second_mapping = adapter.load(torch.tensor([2], dtype=torch.int64, device=device))
    print("after releasing 0, load [2] ->", second_mapping)

    adapter.save(
        torch.tensor([2], dtype=torch.int64, device=device),
        torch.tensor([[200.0, 201.0]], device=device),
    )
    adapter.release(torch.tensor([1, 2], dtype=torch.int64, device=device))

    third_mapping = adapter.load(torch.tensor([2, 3], dtype=torch.int64, device=device))
    print("load [2, 3] ->", third_mapping)
    print("backend snapshot ->", backend.snapshot())
    print("adapter snapshot ->", adapter.debug_snapshot())

    adapter.release(torch.tensor([2, 3], dtype=torch.int64, device=device))
    adapter.shutdown()


if __name__ == "__main__":
    main()
