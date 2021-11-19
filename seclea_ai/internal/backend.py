import multiprocessing
import queue
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional

from seclea_ai.internal.api import Api
from seclea_ai.internal.processors import Sender, Writer
from seclea_ai.internal.threading import DirectorThread, ProcessLoopThread, ProcessorThread


def start_backend(
    settings,
    record_q: Queue,
    result_q: Queue,
    api: Api,
):
    # entrypoint for backend process that starts handler, sender and writer threads.

    # register the exit handler only when called, not on import
    # @atexit.register
    # def handle_exit(*args: "Any") -> None:
    #     logger.info("Internal process exited")

    stop = multiprocessing.Event()
    threads: "List[ProcessLoopThread]" = []

    send_record_q: Queue = queue.Queue()
    record_sender_thread = ProcessorThread(
        processor=Sender,
        name="Sender",
        settings=settings,
        input_q=send_record_q,
        result_q=result_q,
        stop=stop,
        debounce_interval_ms=5000,
        api=api,
    )
    threads.append(record_sender_thread)

    write_record_q: Queue = queue.Queue()
    record_writer_thread = ProcessorThread(
        processor=Writer,
        name="Writer",
        settings=settings,
        input_q=write_record_q,
        result_q=result_q,
        stop=stop,
        debounce_interval_ms=1000,
    )
    threads.append(record_writer_thread)

    record_director_thread = DirectorThread(
        settings=settings,
        input_q=record_q,
        result_q=result_q,
        stop=stop,
        debounce_interval_ms=1000,
        sender_q=send_record_q,
        writer_q=write_record_q,
    )
    threads.append(record_director_thread)

    # maybe add process checking later

    for thread in threads:
        thread.start()

    interrupt_count = 0
    while not stop.is_set():
        try:
            # wait for stop event
            while not stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            interrupt_count += 1
        finally:
            if interrupt_count >= 2:
                stop.set()
        # TODO add logging.

    for thread in threads:
        thread.join()


class Backend(object):
    # multiprocessing context or module
    _multiprocessing: multiprocessing.context.BaseContext
    _internal_pid: int
    backend_process: Optional[multiprocessing.process.BaseProcess]
    _settings: Optional[Dict]
    record_q: Optional["multiprocessing.Queue"]
    result_q: Optional["multiprocessing.Queue"]

    def __init__(self, settings: Dict = None, log_level: int = None) -> None:
        self._done = False
        self.record_q = None
        self.result_q = None
        self.backend_process = None
        self._settings = settings

        self._multiprocessing = multiprocessing  # type: ignore
        self._multiprocessing_setup()

    def _multiprocessing_setup(self) -> None:

        # defaulting to spawn for now, fork needs more testing
        start_method = "spawn"

        ctx = multiprocessing.get_context(start_method)
        self._multiprocessing = ctx

    def ensure_launched(self) -> None:
        """Launch backend worker if not running."""
        settings: Dict[str, Any] = dict(self._settings or ())

        self.record_q = self._multiprocessing.Queue()
        self.result_q = self._multiprocessing.Queue()

        # instantiate Api here so that it definitely has access to the same filesystem
        self.backend_process = self._multiprocessing.Process(
            target=start_backend,
            kwargs=dict(
                settings=settings,
                record_q=self.record_q,
                result_q=self.result_q,
                api=Api(settings=settings),
            ),
        )
        self.backend_process.name = "seclea_internal"

        self.backend_process.start()
        self._internal_pid = self.backend_process.pid

    def cleanup(self) -> None:
        # TODO: make _done atomic
        if self._done:
            return
        self._done = True
        if self.backend_process:
            self.backend_process.join()

        if self.record_q:
            self.record_q.close()
        if self.result_q:
            self.result_q.close()
        # No printing allowed from here until redirect restore!!!
