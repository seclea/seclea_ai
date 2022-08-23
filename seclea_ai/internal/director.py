"""
File storing and uploading data to server
"""
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from itertools import chain
from typing import Dict, Set

from seclea_ai.internal.api.base import BaseModelApi
from seclea_ai.lib.seclea_utils.object_management import Tracked
from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel
from .processors.sender import Sender
from .processors.writer import Writer
from .threading import SingleThreadTaskExecutor

logger = logging.getLogger("seclea_ai")


class DirectorException(Exception):
    pass


class Director:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, cache_dir: str):
        # setup some defaults
        self.writer = Writer(cache_dir=cache_dir)
        self.sender = Sender(cache_dir=cache_dir)
        self.write_threadpool_executor = ThreadPoolExecutor(max_workers=4)
        self.send_thread_executor = SingleThreadTaskExecutor()
        self.send_executing: Set[Future] = set()
        self.write_executing: Set[Future] = set()
        self.errors = list()

    def __del__(self):
        self.terminate()

    def terminate(self) -> None:
        """
        Cleans up resources - usually only called on unscheduled exits.
        :return: None
        """
        # cancel any ongoing work.
        for future in chain(self.send_executing, self.write_executing):
            future.cancel()
        # wait until all resources freed
        self._shutdown_threads()

    def complete(self) -> None:
        """
        Finalises all work and tidies up.
        :return: None
        """
        # TODO: [BUG] if this function is called just after cache_upload_object, objects will be sent twice.
        for future in chain(self.send_executing, self.write_executing):
            future.result()
            self._ensure_error_count_0()
        self._shutdown_threads()

    def cache_upload_object(self, obj_tracked: Tracked, obj_bs: BaseModel, api: BaseModelApi, params: dict):
        self._store_entity(obj_tr=obj_tracked, obj_bs=obj_bs, api=api)
        self._send_entity(api=api, obj_bs=obj_bs, params=params)

    def _store_entity(self, obj_tr: Tracked, obj_bs: BaseModel,
                      api: BaseModelApi) -> None:  # TODO add return for status
        """
        @param obj_tr:
        @param obj_bs:
        @return:
        """
        # check for errors and throw if there are any
        future = self.write_threadpool_executor.submit(self.writer.cache_object, obj_tr=obj_tr, obj_bs=obj_bs, api=api)
        self.write_executing.add(future)
        future.add_done_callback(self._callback_write)

    def _send_entity(self, api: BaseModelApi, obj_bs: BaseModel, params: Dict) -> None:  # TODO add return for status
        """
        Queue an entity for sending (Dataset, DatasetTransformation, TrainingRun or ModelState)
        :param entity_dict: The details needed to send that entity. The same as for storing.
        :return: None
        """
        # check for errors and throw if there are any
        self._ensure_error_count_0()
        # put in queue for sending
        future = self.send_thread_executor.submit(
            self.sender.create_object, api=api, obj_bs=obj_bs, params=params)
        self.send_executing.add(future)
        future.add_done_callback(self._callback_send)

    def _handle_callback(self, future, future_set: Set[Future], succsess_msg: str = "completed - "):
        try:
            result = future.result()
        except Exception as e:
            self.errors.append(e)
        else:
            logger.debug(f"{succsess_msg} {result}")
        finally:
            future_set.remove(future)

    def _callback_write(self, future) -> None:
        self._handle_callback(future, self.write_executing, "write complete - ")

    def _callback_send(self, future) -> None:
        self._handle_callback(future, self.send_executing, "send complete - ")

    def _ensure_error_count_0(self):
        """Check the error queue and throw any errors in there. Needed for user thread to deal with them"""
        if len(self.errors) == 0:
            return
        # make sure the errors are printed so we don't miss any
        for error in self.errors:
            logger.error(error)
        # we can only raise one so we raise the first one in the list.
        raise DirectorException(f"Errors processing {len(self.errors)} requests to director, please check error logs.")

    def _shutdown_threads(self, wait=True, **thread_kwargs):
        self.write_threadpool_executor.shutdown(wait=wait, **thread_kwargs)
        self.send_thread_executor.shutdown(wait=True, **thread_kwargs)

    def try_cleanup(self):
        """
        Tries to clean up any failed sends or records left in local storage.
        TODO complete when in common backend (to avoid conflicts)
        :return: None
        """
        pass
