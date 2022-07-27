import queue
import time
from multiprocessing import Event, Queue
from threading import Thread
from typing import Type

from seclea_ai.internal.processors import Processor


class ProcessLoopThread(Thread):
    def __init__(
        self,
        input_q: Queue,
        started: Event,
        stop: Event,
        complete: Event,
        completed: Event,
        debounce_interval_ms: float,
    ):
        super(ProcessLoopThread, self).__init__()
        self._input_q = input_q
        # self._result_q = result_q
        self._stop_event = stop
        self._complete_event = complete
        self._started_event = started
        self._completed = completed
        self._debounce_interval_ms = debounce_interval_ms

    def _setup(self) -> None:
        raise NotImplementedError

    def _process(self, item) -> None:
        raise NotImplementedError

    def _complete(self) -> None:
        raise NotImplementedError

    def _terminate(self) -> None:
        raise NotImplementedError

    def _debounce(self) -> None:
        raise NotImplementedError

    def run(self):
        self._setup()
        self._started_event.set()
        start = time.time()
        while not self._stop_event.is_set():
            if time.time() - start >= self._debounce_interval_ms / 1000.0:
                self._debounce()
                start = time.time()
            try:
                record = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue
            self._process(record)
        if self._complete_event.is_set():
            self._complete()
        self._terminate()


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


class ProcessorThread(ProcessLoopThread):
    def __init__(
        self,
        processor: Type[Processor],
        name,
        settings,
        input_q: Queue,
        # result_q: Queue,
        started: Event,
        stop: Event,
        complete: Event,
        completed: Event,
        debounce_interval_ms: float,
        **kwargs,
    ):
        super(ProcessorThread, self).__init__(
            input_q=input_q,
            started=started,
            stop=stop,
            complete=complete,
            completed=completed,
            debounce_interval_ms=debounce_interval_ms,
        )
        self._processor_class = processor  # just a class here - instantiate and populate in _setup.
        self._settings = settings
        self.name = name
        self._completed = completed
        self._kwargs = kwargs

    def _setup(self) -> None:
        self._processor: Processor = self._processor_class(
            settings=self._settings,
            input_q=self._input_q,
            completed=self._completed,
            # result_q=self._result_q,
            **self._kwargs,
        )

    def _process(self, record) -> None:
        self._processor.handle(record)

    def _complete(self) -> None:
        self._processor.complete()

    def _terminate(self) -> None:
        self._processor.terminate()

    def _debounce(self) -> None:
        # do nothing for now, add debounce later
        return
