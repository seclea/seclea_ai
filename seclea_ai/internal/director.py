from multiprocessing import Event, Queue


class Director:

    _input_q: Queue
    _result_q: Queue
    _stop: Event
    _sender_q: Queue
    _writer_q: Queue

    def __init__(
        self,
        settings,
        input_q: Queue,
        result_q: Queue,
        stop: Event,
        sender_q: Queue,
        writer_q: Queue,
    ):
        self._settings = settings
        self._input_q = input_q
        self._result_q = result_q
        self._stop = stop
        self._sender_q = sender_q
        self._writer_q = writer_q

    def __len__(self):
        return self._input_q.qsize()

    def handle(self, record):
        # TODO check record type (if we have different ones)
        self._dispatch_record(record)

    def _dispatch_record(self, record):
        # default is to write to file if not connected to the internet.
        # TODO add congestion handling - write to file if sending q too long etc.
        # send files from storage to send when congestion reduced. - use flag and watcher?
        if not self._settings["offline"]:
            self._sender_q.put(record)
        else:
            self._writer_q.put(record)

    def handle_request_shutdown(self):
        # TODO add request and result
        self._stop.set()

    def terminate(self):
        # for now nothing, normally clean up of some kind
        pass

    # lots of thread/process handling logic to add here.
