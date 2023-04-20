"""
File storing and uploading data to server
"""
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, List, Callable

from peewee import SqliteDatabase
from pympler import asizeof

from .api.api_interface import Api
from .exceptions import (
    APIError,
    AuthenticationError,
    RequestTimeoutError,
    ServiceDegradedError,
    StorageSpaceError,
)
from ..internal.persistence.record import Record, RecordStatus
from .processors.sender import Sender
from .processors.writer import Writer
from .threading import SingleThreadTaskExecutor

logger = logging.getLogger("seclea_ai")


class Director:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, api, db):
        # setup some defaults
        self._settings = settings
        self._api: Api = api
        self._db: SqliteDatabase = db
        self.writer = Writer(settings=settings)
        self.sender = Sender(settings=settings, api=api)
        self.write_threadpool_executor = ThreadPoolExecutor(max_workers=4)
        self.send_thread_executor = SingleThreadTaskExecutor()
        self.send_executing: Dict[Future, Dict] = dict()
        self.write_executing: List[Future] = list()
        self.errors = list()

    def __del__(self):
        try:
            self.terminate()
        except AttributeError:
            # means we already removed references - cannot do anything.
            pass

    def terminate(self) -> None:
        """
        Cleans up resources - usually only called on unscheduled exits.
        :return: None
        """
        # cancel any ongoing work.
        for future in self.send_executing.keys():
            future.cancel()
        for future in self.write_executing:
            future.cancel()
        # wait until all resources freed
        self.write_threadpool_executor.shutdown(wait=True)
        self.send_thread_executor.shutdown(wait=True)

    def complete(self) -> None:
        """
        Finalises all work and tidies up.
        :return: None
        """
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
        """
        Queue an entity for storing (Dataset, DatasetTransformation, TrainingRun or ModelState)
        :param entity_dict: The details needed to store that entity. The same as for sending.
        :return: None
        """
        # check for errors and throw if there are any
        self._check_and_throw()
        self._check_resources(entity_dict=entity_dict)
        future = self.write_threadpool_executor.submit(
            self.writer.funcs[entity_dict["entity"]], **entity_dict
        )
        self.write_executing.append(future)
        future.add_done_callback(self._write_completed)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        """
        Queue an entity for sending (Dataset, DatasetTransformation, TrainingRun or ModelState)
        :param entity_dict: The details needed to send that entity. The same as for storing.
        :return: None
        """
        # check for errors and throw if there are any
        self._check_and_throw()
        # put in queue for sending
        future = self.send_thread_executor.submit(
            self.sender.funcs[entity_dict["entity"]], **entity_dict
        )
        self.send_executing[future] = entity_dict
        future.add_done_callback(self._send_completed)

    def try_cleanup(self):
        """
        Tries to clean up any failed sends or records left in local storage.
        TODO complete when in common backend (to avoid conflicts)
        :return: None
        """
        pass

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

    def _check_resources(self, entity_dict):
        # get size in memory
        if entity_dict.get("dataset", None) is not None:
            size = asizeof.asizeof(entity_dict["dataset"])
            stored_size = size / 10
        elif entity_dict.get("model", None) is not None:
            size = asizeof.asizeof(entity_dict["model"])
            stored_size = size * 50
        else:
            return

        # get currently used space from db
        self._db.connect()
        records = Record.select().where(Record.status == RecordStatus.STORED)
        current_stored = 0
        for record in records:
            current_stored += record.size
        self._db.close()

        # break on storage overflow
        if stored_size + current_stored > self._settings["max_storage_space"]:
            raise StorageSpaceError("Specified storage size exceeded")

        # limit is in director (from settings)
        # NOTE not breaking on memory as reliability of estimating memory impact is too low
        # and the fact that swap is common reduces the severity of the impact.
