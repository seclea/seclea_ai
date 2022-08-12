"""
File storing and uploading data to server
"""
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Callable

from .exceptions import (
    APIError,
    AuthenticationError,
    RequestTimeoutError,
    ServiceDegradedError,
)
from .processors.sender import Sender
from .processors.writer import Writer
from .threading import SingleThreadTaskExecutor

logger = logging.getLogger("seclea_ai")


class Director:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, api):
        # setup some defaults
        self._settings = settings
        self._api = api
        self.writer = Writer(settings=settings)
        self.sender = Sender(settings=settings, api=api)
        self.write_threadpool_executor = ThreadPoolExecutor(max_workers=4)
        self.send_thread_executor = SingleThreadTaskExecutor()
        self.send_executing: Dict[Future, Dict] = dict()
        self.write_executing: List[Future] = list()
        self.errors = list()

    def __del__(self):
        self.terminate()

    def terminate(self):
        # cancel any ongoing work.
        for future in self.send_executing.keys():
            future.cancel()
        for future in self.write_executing:
            future.cancel()
        # wait until all resources freed
        self.write_threadpool_executor.shutdown(wait=True)
        self.send_thread_executor.shutdown(wait=True)

    def complete(self):
        # wait for the writes to complete
        for future in self.write_executing:
            future.result()
            self._check_and_throw()
        # wait for the last send to complete
        for future in self.send_executing.keys():
            if self.send_executing[future] is not None:
                future.result()
                self._check_and_throw()
        self.write_threadpool_executor.shutdown(wait=True)
        self.send_thread_executor.shutdown(wait=True)

    def store_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        # check for errors and throw if there are any
        self._check_and_throw()
        future = self.write_threadpool_executor.submit(
            self.writer.funcs[entity_dict["entity"]], **entity_dict
        )
        self.write_executing.append(future)
        future.add_done_callback(self._write_completed)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        # check for errors and throw if there are any
        self._check_and_throw()
        # put in queue for sending
        future = self.send_thread_executor.submit(
            self.sender.funcs[entity_dict["entity"]], **entity_dict
        )
        self.send_executing[future] = entity_dict
        future.add_done_callback(self._send_completed)

    def _write_completed(self, future) -> None:
        try:
            result = future.result()
        except Exception as e:
            self.errors.append(e)
        else:
            logger.debug(f"Write completed - {result}")
        finally:
            self.write_executing.remove(future)

    def _send_completed(self, future) -> None:
        # here we use a queue to guarantee the ordering and minimise dependency issues.
        try:
            result = future.result()
        except AuthenticationError:
            self._try_resend_with_action(future=future, function=self._api.authenticate)
        except RequestTimeoutError:
            self._try_resend_with_action(future=future)
        except ServiceDegradedError:
            self._try_resend_with_action(future=future)
        # this is only caught if more specific doesn't catch which means it is something we can't deal with.
        except Exception as e:
            self.errors.append(e)
            return  # need to return otherwise second try except block runs.
        else:
            logger.debug(f"Send completed - {result}")
        finally:
            self.send_executing[future] = None

    def _check_and_throw(self):
        """Check the error queue and throw any errors in there. Needed for user thread to deal with them"""
        if len(self.errors) == 0:
            return
        # make sure the errors are printed so we don't miss any
        for error in self.errors:
            logger.error(error)
        # we can only raise one so we raise the first one in the list.
        raise self.errors[0]

    def _try_resend_with_action(self, future: Future, function: Callable = None) -> None:
        """Try the request again with some action - only one retry"""
        entity_dict = self.send_executing[future]
        try:
            if entity_dict["break"]:
                raise APIError("Authentication failed too many times")
        except KeyError:
            entity_dict["break"] = True
            if function is not None:
                function()
            new_future = self.send_thread_executor.submit_first(
                self.sender.funcs[entity_dict["entity"]], **entity_dict
            )
            self.send_executing[new_future] = entity_dict
            new_future.add_done_callback(self._send_completed)
