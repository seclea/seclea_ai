import abc
import os
import uuid
from abc import ABC
from multiprocessing import Queue
from typing import Dict, List

import pandas as pd

from seclea_ai.internal.api import Api
from seclea_ai.storage import Storage


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):

    _input_q: Queue
    _result_q: Queue

    def __init__(self, settings, input_q: Queue, result_q: Queue, **kwargs):
        self._settings = settings
        self._input_q = input_q
        self._result_q = result_q

    def __len__(self):
        return self._input_q.qsize()

    @abc.abstractmethod
    def handle(self, record):
        pass

    @abc.abstractmethod
    def terminate(self):
        pass


class Writer(Processor):

    _input_q: Queue
    _result_q: Queue

    def __init__(self, settings, input_q: Queue, result_q: Queue):
        super(Writer, self).__init__(settings=settings, input_q=input_q, result_q=result_q)
        self._settings = settings
        self._input_q = input_q
        self._result_q = result_q
        self._db = Storage(root=self._settings["project_root"], db_name="runs.db")

    def handle(self, record) -> None:
        # TODO check record type (if we have different ones)
        self._write(record)

    def _write(self, record) -> None:
        """
        Writes a record to local storage. This is intended primarily as a buffer and backup if internet
        connectivity is not available.
        Record must contain:
            {
                username: str,
                project_id: int,
                entity_id: int, TODO something better that handles local uniqueness and preserves order for upload.
            }

        :param record:
        :return:
        """
        key = _assemble_key(record)
        self._db.write(key, record)

    def terminate(self) -> None:
        pass


class Sender(Processor):
    _input_q: Queue
    _result_q: Queue
    _publish_q: Queue

    def __init__(self, settings, input_q: Queue, result_q: Queue, api: Api):
        super(Sender, self).__init__(settings=settings, input_q=input_q, result_q=result_q)
        self._settings = settings
        self._input_q = input_q
        self._result_q = result_q
        self._api = api

    def handle(self, record) -> None:
        # TODO check record type (if we have different ones)
        self._send_training_run(record)

    def _send_training_run(self, record) -> None:
        # send the record and put the response in the result_q
        """
        Assume record contains everything about the training_run
        {
            name: str,
            model_class: str,
            dataset: int,
            params: dict,
            model_states: list[filenames?],
        }
        :param record:
        :return:
        """
        pass

    def _send_dataset(self, record) -> None:
        # send the record and put the response in the result_q
        """
        Assume record contains everything about the dataset
        {
            name: str,
            metadata: dict,
            parent: Optional[int],
            transformations: List[Callable],
            files: List[Pathlike]
        }
        For now just send directly in here, may add async processing using threads later depending on requirements.
        :param record:
        :return:
        """
        self._api.reauthenticate()
        temp_dataset_path = self._aggregate_dataset(datasets=record["files"])
        # TODO need to handle large metadata - will need to send empty with file and use update request to fill
        # it in in a second request.
        query_params = self.get_dataset_query_params(record)
        response = self._api.transport.send_file(
            url_path=self._api.dataset_endpoint,
            file_path=temp_dataset_path,
            query_params=query_params,
        )
        # clean up temp file
        os.remove(temp_dataset_path)
        self._result_q.put(response)

    def terminate(self) -> None:
        # nothing to do here for now
        pass

    def _aggregate_dataset(self, datasets: List[str]) -> str:
        """
        Aggregates a list of dataset paths into a single file for upload.
        NOTE the files must be split by row and have the same format otherwise this will fail or cause unexpected format
        issues later.
        :param datasets:
        :return:
        """
        loaded_datasets = [pd.read_csv(dset) for dset in datasets]
        aggregated = pd.concat(loaded_datasets, axis=0)
        if not os.path.exists(self._settings["cache_dir"]):
            os.makedirs(self._settings["cache_dir"])
        # save aggregated and return path as string
        aggregated.to_csv(
            os.path.join(self._settings["cache_dir"], f"temp_dataset_{uuid.uuid4()}.csv"),
            index=False,
        )
        return os.path.join(self._settings["cache_dir"], "temp_dataset.csv")

    def get_dataset_query_params(self, record) -> Dict:
        """
        Get query_params for the dataset from the information in the record.
        TODO remove this by having better defined classes in the package - may coincide with using protobuf.
        :param record:
        :return: Dict The query params dict
        """
        return {
            "project": self._settings["project"],
            "name": record["name"],
            "metadata": record["metadata"],
            "parent": record["parent"],
            "transformations": record["transformations"],
        }
