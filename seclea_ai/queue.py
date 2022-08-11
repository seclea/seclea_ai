import queue
from queue import Queue
from time import monotonic as time


class PreemptableQueue(Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

    def _put_first(self, item):
        self.queue.appendleft(item)

    def put_first(self, item, block=True, timeout=None):
        """Put an item into front of the queue.
        If optional args 'block' is true and 'timeout' is None (the default),
        block if necessary until a free slot is available. If 'timeout' is
        a non-negative number, it blocks at most 'timeout' seconds and raises
        the Full exception if no free slot was available within that time.
        Otherwise ('block' is false), put an item on the queue if a free slot
        is immediately available, else raise the Full exception ('timeout'
        is ignored in that case).
        """
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise queue.Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise queue.Full
                        self.not_full.wait(remaining)
            self._put_first(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def put_first_nowait(self, item):
        """Put an item into front of the queue without blocking.
        Only enqueue the item if a free slot is immediately available.
        Otherwise raise the Full exception."""
        return self.put_first(item, block=False)
