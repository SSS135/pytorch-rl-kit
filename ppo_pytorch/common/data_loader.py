from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from typing import List


class DataLoader:
    def __init__(self, data, chunks, device, num_threads, dim=0):
        self.data = data
        self.chunks = chunks
        self.device = device
        self.num_threads = num_threads
        self.dim = dim
        self._chunk_index = 0
        self._executor = ThreadPoolExecutor(max_workers=num_threads)
        self._futures: List[Future] = []
        for _ in range(min(len(chunks), num_threads)):
            self._submit()
        assert len(self._futures) > 0, (len(chunks), num_threads)

    def get_next_batch(self):
        future = self._futures[0]
        self._futures.remove(future)
        batch = future.result()
        if self._chunk_index < len(self.chunks):
            self._submit()
        return batch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown()

    def _submit(self):
        self._futures.append(self._executor.submit(self._load_chunk, self._chunk_index))
        self._chunk_index += 1

    def _load_chunk(self, index):
        chunk = self.chunks[index]
        chunk = *([slice(None)] * self.dim), chunk

        def extract_chunk(x):
            return x[chunk] if x.device == self.device else x[chunk].to(self.device)

        if isinstance(self.data, dict):
            return {k: extract_chunk(v) for k, v in self.data.items()}
        else:
            return [extract_chunk(x) for x in self.data]