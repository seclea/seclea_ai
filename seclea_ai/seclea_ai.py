"""
Description for seclea_ai.py
"""
import inspect
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Union

import pandas as pd
from requests import Response
from seclea_utils.core import CompressedFileManager, RequestWrapper, Zstd, decode_func, encode_func

from seclea_ai.authentication import AuthenticationService
from seclea_ai.model_management import get_model_manager


def handle_response(res: Response, msg):
    if not res.ok:
        print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")


class SecleaAI:
    def __init__(
        self,
        project_name,
        framework,
        plat_url="https://platform.seclea.com",
        auth_url="https://auth.seclea.com",
    ):
        self._model_manager = get_model_manager(
            framework, CompressedFileManager(compression=Zstd())
        )
        self._auth_service = AuthenticationService(RequestWrapper(auth_url))
        self._transmission = RequestWrapper(server_root_url=plat_url)
        self._transmission.headers = self._auth_service.handle_auth()
        self._project = None
        self._project_name = project_name
        self._models = None
        self._model = None
        self._model_name = None
        self._model_framework = None
        self._frameworks = {"sklearn"}
        self._dataset = None
        self._training_run = None
        self._training_runs = None
        self._cache_dir = os.path.join(Path.home(), f".seclea/{self._project_name}")
        self._setup_project(project_name=project_name)

    def _setup_project(self, project_name):
        """
        Sets up a project.
        Checks if it exists and if it does gets any datasets or models associated with it and the latest training_run id.
        If it doesn't exist it creates it and uploads it.
        :return: None
        """
        # here check the project exists and call create if not.
        res = self._transmission.get("/collection/projects", query_params={"name": project_name})
        if res.status_code == 200 and len(res.json()) > 0:
            self._project = res.json()[0]["id"]
            # setup the models and datasets available.
        else:
            proj_res = self._create_project()
            if proj_res.status_code == 201:
                try:
                    self._project = proj_res.json()["id"]
                except KeyError:
                    print(f"There was an issue: {proj_res.text}")
                    resp = self._transmission.get(
                        url_path="/collection/projects",
                        query_params={
                            "name": project_name,
                        },
                    )
                    self._project = resp.json()[0]["id"]
            handle_response(res, "Some issue with creating the project")
        model_res = self._transmission.get("/collection/models")
        self._models = model_res.json()
        dataset_res = self._transmission.get(
            "/collection/datasets", query_params={"project": self._project}
        )
        self._datasets = dataset_res.json()

    def login(self) -> None:
        """
        Override login, this also overwrites the stored credentials in ~/.seclea/config.
        Note. In some circumstances the password will be echoed to stdin. This is not a problem in Jupyter Notebooks
        but may appear in scripting usage.

        :return: None

        Example::

            >>> seclea = SecleaAI(project_name="Test Project")
            >>> seclea.login()
        """
        self._transmission.headers = self._auth_service.login()

    def init_project(self, model_name: str, framework: str, dataset_name: str) -> None:
        """
        Wrapper or shortcut method that initializes the project. Sets model and dataset.
        Throws exception if dataset has not been uploaded.

        :param model_name: The name of the model.

        :param framework: The framework being used. Currently only "sklearn" is supported.

        :param dataset_name: The name of the dataset.

        :return: None

        Example::

            >>>
        """
        self.set_model(model_name, framework)
        self.set_dataset(dataset_name)

    def set_model(self, model_name: str, framework: str) -> None:
        """
        Set the model for this session.
        Checks if it has already been uploaded. If not it will upload it.

        :param model_name: The name for the architecture/algorithm. eg. "GradientBoostedMachine" or "3-layer CNN".

        :param framework: The machine learning framework being used. eg. "sklearn" or "pytorch"

        :return: None

        Example::

            >>>
        """
        if framework not in self._frameworks:
            raise ValueError(f"Framework must be one of {self._frameworks}")
        # check if the model is in those already uploaded.
        for model in self._models:
            if model["name"] == model_name and model["framework"] == framework:
                self._model = model["id"]
                return
        # if we got here that means that the model has not been uploaded yet. So we upload it.
        res = self._upload_model(model_name=model_name, framework=framework)
        if res.status_code == 201:
            try:
                self._model = res.json()["id"]
            except KeyError:
                pass
        else:
            print(f"There was an issue: {res.text}")
            resp = self._transmission.get(
                url_path="/collection/models",
                query_params={
                    "name": model_name,
                    "framework": framework,
                },
            )
            self._model = resp.json()[0]["id"]
        handle_response(res, "Some issue with setting the model")

    def set_dataset(self, dataset_name: str) -> None:
        """
        Set the dataset for the session.
        Checks if it has been uploaded, if not throws an Exception

        :param dataset_name: The name of the dataset.

        :return: None

        Example::

            >>>
        """
        for dataset in self._datasets:
            if dataset["name"] == dataset_name:
                self._dataset = dataset["id"]
                return
        # if we got here then the dataset has not been uploaded somehow so the user needs to do so.
        raise Exception(  # TODO replace with custom or more appropriate Exception.
            "The dataset has not been uploaded yet, please use upload_dataset(path, id, metadata) to upload one."
        )

    def upload_dataset(self, dataset: Union[str, List[str]], dataset_name: str, metadata: Dict):
        """
        Uploads a dataset. Does not set the dataset for the session. Should be carried out before setting the dataset.

        :param dataset: Path or list of paths to the dataset. If a list then they must be split by row only and all
            files must contain column names as a header line.

        :param dataset_name: The name of the dataset.

        :param metadata: Any metadata about the dataset.

        :return: None

        Example::

            >>>
        """
        temp = False
        if self._project is None:
            raise Exception("You need to create a project before uploading a dataset")
        if isinstance(dataset, List):
            dataset = self._aggregate_dataset(dataset)
            temp = True
        dataset_queryparams = {
            "project": self._project,
            "name": dataset_name,
            "metadata": json.dumps(metadata),
        }
        res = self._transmission.send_file(
            url_path="/collection/datasets",
            file_path=dataset,
            query_params=dataset_queryparams,
        )
        if res.status_code == 201:
            self._datasets.append(res.json())
            if temp:
                os.remove(dataset)
        handle_response(res, "Error uploading dataset: ")

    def upload_training_run(self, model, transformations: List[Callable]):
        """
        Takes a model and extracts the necessary data for uploading the training run.

        :param model: An sklearn Estimator model.

        :param transformations: A list of functions that preprocess the Dataset.

        :return: None

        Example::

            >>>
        """
        # if we haven't requested the training runs for this model do that.
        if self._training_runs is None:
            training_runs_res = self._transmission.get(
                "/collection/training-runs",
                query_params={"project": self._project, "model": self._model},
            )
            self._training_runs = training_runs_res.json()

        # Create the training run name
        largest = -1
        for training_run in self._training_runs:
            num = int(training_run["name"].split(" ")[2])
            if num > largest:
                largest = num
        training_run_name = f"Training Run {largest + 1}"

        # extract params from the model
        params = model.get_params()  # TODO make compatible with other frameworks.

        # upload training run
        tr_res = self._upload_training_run(training_run_name=training_run_name, params=params)
        # if the upload was successful, add the new training_run to the list to keep the names updated.
        if tr_res.status_code == 201:
            self._training_run = tr_res.json()["id"]
            self._training_runs.append(tr_res.json())

        # upload transformations.
        trans_resps = self._upload_transformations(
            transformations=transformations, training_run_id=self._training_run
        )
        for trans_res in trans_resps:
            if trans_res.status_code != 201:
                handle_response(trans_res, "There was an issue with uploading the transformations.")

        # upload model state. TODO figure out how this fits with multiple model states.
        self._upload_model_state(
            model=model, training_run_id=self._training_run, sequence_num=0, final=True
        )

    def _create_project(self):
        """
        Creates a new project.
        :return:
        """
        res = self._transmission.send_json(
            url_path="/collection/projects",
            obj={
                "name": self._project_name,
                "description": "Please add a description..",
            },
        )
        return res

    def _upload_model(self, model_name: str, framework: str):
        """

        :param model_name:
        :param framework:
        :return:
        """
        res = self._transmission.send_json(
            url_path="/collection/models",
            obj={
                "name": model_name,
                "framework": framework,
            },
        )
        return res

    def _upload_training_run(self, training_run_name: str, params: Dict):
        """

        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyper parameters of the model - can auto extract?
        :return:
        """
        if self._project is None:
            raise Exception("You need to create a project before uploading a training run")
        res = self._transmission.send_json(
            url_path="/collection/training-runs",
            obj={
                "project": self._project,
                "dataset": self._dataset,
                "model": self._model,
                "name": training_run_name,
                "params": params,
            },
        )
        return res

    def _upload_model_state(self, model, training_run_id: int, sequence_num: int, final: bool):
        os.makedirs(
            os.path.join(self._cache_dir, str(training_run_id)),
            exist_ok=True,
        )

        save_path = self._model_manager.save_model(
            model,
            os.path.join(
                Path.home(), f".seclea/{self._project_name}/{training_run_id}/model-{sequence_num}"
            ),
        )

        res = self._transmission.send_file(
            url_path="/collection/model-states",
            file_path=save_path,
            query_params={
                "sequence_num": sequence_num,
                "training_run": training_run_id,
                "final_state": final,
            },
        )
        if res.status_code == 201:
            os.remove(save_path)
        return res

    def _upload_transformations(self, transformations: List[Callable], training_run_id: int):
        responses = list()
        for idx, trans in enumerate(transformations):
            data = {
                "name": trans.__name__,
                "code_raw": inspect.getsource(trans),
                "code_encoded": encode_func(trans),
                "order": idx,
                "training_run": training_run_id,
            }
            responses.append(
                self._transmission.send_json(
                    url_path="/collection/dataset-transformations", obj=data
                )
            )

        return responses

    def _load_transformations(self, training_run_id: int):
        """
        Expects a list of code_encoded as set by upload_transformations.
        """
        res = self._transmission.get(
            url_path="/collection/dataset-transformations",
            query_params={"training_run": training_run_id},
        )
        transformations = list(map(lambda x: x["code_encoded"], res.json()))
        return list(map(decode_func, transformations))

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
        # save aggregated and return path as string
        aggregated.to_csv(os.path.join(self._cache_dir, "temp_dataset.csv"), index=False)
        return os.path.join(self._cache_dir, "temp_dataset.csv")
