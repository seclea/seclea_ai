import logging
import queue
import threading
from concurrent.futures import Future, Executor
from typing import Callable, Any

from ..queue import PreemptableQueue

logger = logging.getLogger("seclea_ai")

# TODO add shutdown and exit handler?


def _worker(input_q: PreemptableQueue):
    try:
        while True:
            task = input_q.get(block=True)
            if task is not None:
                task.run()
                del task
            else:
                return
    except BaseException:
        logger.critical("Exception in thread")


class Task:
    def __init__(self, future: Future, func, args, kwargs):
        self.future = future
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        if not self.future.set_running_or_notify_cancel():
            return

        try:
            result = self.func(*self.args, **self.kwargs)
        except BaseException as exc:
            self.future.set_exception(exc)
            # Break a reference cycle with the exception 'exc'
            self = None
        else:
            self.future.set_result(result)


class SingleThreadTaskExecutor(Executor):
    def __init__(self):
        self.queue = PreemptableQueue()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self.thread = threading.Thread(target=_worker, args=[self.queue], daemon=True)
        self.thread.start()

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            w = Task(f, fn, args, kwargs)

            self.queue.put(w)
            return f

    def submit_first(self, fn: Callable, *args: Any, **kwargs: Any) -> Future:
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            w = Task(f, fn, args, kwargs)

            self.queue.put_first(w)
            return f

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._shutdown_lock:
            self._shutdown = True
            self.queue.put(None)
            if cancel_futures:
                # Drain all work items from the queue, and then cancel their
                # associated futures.
                while True:
                    try:
                        work_item = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    if work_item is not None:
                        work_item.future.cancel()
        if wait:
            self.thread.join()


# TODO decide what to do with this block - probably will use when extracting the internal stuff into
# a background process.
# class DirectorThread(ProcessLoopThread):
#
#     _input_q: Queue
#     _result_q: Queue
#     _stop: Event
#
#     def __init__(
#         self,
#         settings,
#         input_q: Queue,
#         result_q: Queue,
#         stop: Event,
#         debounce_interval_ms: float,
#         sender_q,
#         writer_q,
#     ):
#         super(DirectorThread, self).__init__(
#             input_q=input_q, stop=stop, debounce_interval_ms=debounce_interval_ms
#         )
#         self._settings = settings
#         self.name = "DirectorThread"
#         self._sender_q = sender_q
#         self._writer_q = writer_q
#
#     def _setup(self) -> None:
#         self._director = Director(
#             settings=self._settings,
#             input_q=self._input_q,
#             result_q=self._result_q,
#             stop=self._stop,
#             sender_q=self._sender_q,
#             writer_q=self._writer_q,
#         )
#
#     def _process(self, record) -> None:
#         self._director.handle(record)
#
#     def _terminate(self) -> None:
#         self._director.terminate()
#
#     def _debounce(self) -> None:
#         # do nothing for now, add debounce later
#         return
