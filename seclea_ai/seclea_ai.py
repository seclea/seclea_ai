"""
Description for seclea_ai.py
"""
import asyncio
import copy
import os
import threading
from queue import Queue
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from seclea_ai.authentication import AuthenticationService
from seclea_ai.exceptions import AuthenticationError
from seclea_ai.internal.api import Api, handle_response
from seclea_ai.internal.backend import Backend
from seclea_ai.internal.file_processor import FileProcessor
from seclea_ai.lib.seclea_utils.core import RequestWrapper, decode_func
from seclea_ai.lib.seclea_utils.model_management.get_model_manager import ModelManagers
from seclea_ai.transformations import DatasetTransformation


class SecleaAI:
    def __init__(
        self,
        project_name: str,
        organization: str,
        project_root: str = ".",
        platform_url: str = "https://platform.seclea.com",
        auth_url: str = "https://auth.seclea.com",
        username: str = None,
        password: str = None,
    ):
        """
        Create a SecleaAI object to manage a session. Requires a project name and framework.

        :param project_name: The name of the project.

        :param organization: The name of the project's organization.

        :param project_root: The path to the root of the project. Default: "."

        :param platform_url: The url of the platform server. Default: "https://platform.seclea.com"

        :param auth_url: The url of the auth server. Default: "https://auth.seclea.com"

        :param username: seclea username

        :param password: seclea password

        :return: SecleaAI object

        :raises: ValueError - if the framework is not supported.

        Example::

            >>> seclea = SecleaAI(project_name="Test Project", project_root=".")
        """
        self._settings = {
            "project": project_name,
            "project_root": project_root,
            "platform_url": platform_url,
            "auth_url": auth_url,
            "cache_dir": os.path.join(project_root, ".seclea/cache"),
            "offline": False,
        }
        self._backend = Backend(settings=self._settings)
        self._backend.ensure_launched()
        self._auth_service = AuthenticationService(auth_url, RequestWrapper(auth_url))
        self._transmission = RequestWrapper(server_root_url=platform_url)
        # if username is not None and password is not None:
        #     print("Unsecure login")
        #     self.login(username=username, password=password)
        # self._auth_service.authenticate(self._transmission)
        self._auth_service.authenticate(self._transmission, username=username, password=password)
        self._api = Api(self._settings)
        self._project = None
        self._project_name = project_name
        self._organization = organization
        self._available_frameworks = {"sklearn", "xgboost", "lightgbm"}
        self._training_run = None
        self._init_project(project_name=project_name)
        self._file_processor = FileProcessor(
            self._project_name,
            self._organization,
            self._transmission,
            self._api,
            self._auth_service,
        )
        self._storage_q = Queue()
        print("success")

    def login(self, username=None, password=None) -> None:
        """
        Override login, this also overwrites the stored credentials in ~/.seclea/config.
        Note. In some circumstances the password will be echoed to stdin. This is not a problem in Jupyter Notebooks
        but may appear in scripting usage.

        :return: None

        Example::

            >>> seclea = SecleaAI(project_name="Test Project")
            >>> seclea.login()
        """
        success = False
        for i in range(3):
            try:
                self._auth_service.authenticate(
                    self._transmission, username=username, password=password
                )
                success = True
                break
            except AuthenticationError as e:
                print(e)
        if not success:
            raise AuthenticationError("Failed to login.")

    def upload_dataset_split(
        self,
        X: Union[DataFrame, np.ndarray],
        y: Union[DataFrame, np.ndarray],
        dataset_name: str,
        metadata: Dict,
        transformations: List[DatasetTransformation] = None,
    ) -> None:
        """
        Uploads a dataset.

        :param X: DataFrame The samples of the Dataset.

        :param y: Dataframe The labels of the Dataset

        :param dataset_name: The name of the Dataset

        :param metadata: Any metadata about the Dataset. Required keys are:
            "index" and "continuous_features"

        :param transformations: A list of DatasetTransformation's.

                        If your Dataset is large try and call this function more often with less DatasetTransformations
                        as the function currently requires (no. DatasetTransformations x Dataset size) memory.

                        See DatasetTransformation for more details.

        :return: None
        """
        dataset = self._assemble_dataset({"X": X, "y": y})
        # potentially fragile vvv TODO check this vvv
        metadata["outcome_name"] = y.name
        # try and extract the index automatically
        try:
            metadata["index"] = X.index.name
        except AttributeError:
            metadata["index"] = 0
        self.upload_dataset(dataset, dataset_name, metadata, transformations)

    def upload_dataset(
        self,
        dataset: Union[str, List[str], DataFrame],
        dataset_name: str,
        metadata: Dict,
        transformations: List[DatasetTransformation] = None,
    ) -> None:
        """
        Uploads a dataset.

        :param dataset: DataFrame, Path or list of paths to the dataset.
            If a list then they must be split by row only and all
            files must contain column names as a header line.

        :param dataset_name: The name of the dataset.

        :param metadata: Any metadata about the dataset. Note that if using a Path or list of Paths then if there is an
            index that you use when loading the data, it must be specified in the metadata.

        :param transformations: A list of DatasetTransformation's.

                        If your Dataset is large try call this function more often with less DatasetTransformations
                        as the function currently requires no. DatasetTransformations x Dataset size memory to function.

                        See DatasetTransformation for more details.

        :return: None

        Example:: TODO update docs
            >>> seclea = SecleaAI(project_name="Test Project")
            >>> dataset = pd.read_csv("/test_folder/dataset_file.csv")
            >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
            >>> seclea.upload_dataset(dataset=dataset, dataset_name="Multifile Dataset", metadata=dataset_metadata)

        Example with file::

            >>> seclea.upload_dataset(dataset="/test_folder/dataset_file.csv", dataset_name="Test Dataset", metadata={})
            >>> seclea = SecleaAI(project_name="Test Project", organization="Test Organization")

        Assuming the files are all in the /test_folder/dataset directory.
        Example with multiple files::

            >>> files = os.listdir("/test_folder/dataset")
            >>> seclea = SecleaAI(project_name="Test Project")
            >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
            >>> seclea.upload_dataset(dataset=files, dataset_name="Multifile Dataset", metadata=dataset_metadata)


        """
        # processing the final dataset - make sure it's a DataFrame
        if self._project is None:
            raise Exception("You need to create a project before uploading a dataset")

        if isinstance(dataset, List):
            dataset = self._aggregate_dataset(dataset, index=metadata["index"])
        elif isinstance(dataset, str):
            dataset = pd.read_csv(dataset, index_col=metadata["index"])

        dataset_hash = pd.util.hash_pandas_object(dataset).sum()

        if transformations is not None:

            upload_queue = self._generate_intermediate_datasets(
                transformations=transformations,
                dataset_name=dataset_name,
                dataset_hash=dataset_hash,
                user_metadata=metadata,
            )

            # check for duplicates
            for up_kwargs in upload_queue:
                if (
                    pd.util.hash_pandas_object(up_kwargs["dataset"]).sum()
                    == up_kwargs["parent_hash"]
                ):
                    raise AssertionError(
                        f"""The transformation {up_kwargs['transformation'].func.__name__} does not change the dataset.
                        Please remove it and try again."""
                    )
            # upload all the datasets and transformations.
            for up_kwargs in upload_queue:
                # self._upload_dataset(**up_kwargs)  # TODO change
                self._storage_q.put(
                    {
                        "function": "upload_dataset",
                        "project": self._project,
                        "dataset": up_kwargs["dataset"],
                        "dataset_name": up_kwargs["dataset_name"],
                        "metadata": up_kwargs["metadata"],
                        "parent_hash": up_kwargs["parent_hash"],
                        "transformation": up_kwargs["transformation"],
                    }
                )
            return

        # this only happens if this has no transformations ie. it is a Raw Dataset.
        self._storage_q.put(
            {
                "function": "upload_dataset",
                "project": self._project,
                "dataset": dataset,
                "dataset_name": dataset_name,
                "metadata": metadata,
                "parent_hash": None,
                "transformation": None,
            }
        )
        _thread = threading.Thread(target=self._file_processor.writer(self._storage_q))
        _thread.start()
        _thread.join()

        _sending_thread = threading.Thread(target=self._file_processor.sender())
        _sending_thread.start()
        _sending_thread.join()

    def _generate_intermediate_datasets(
        self, transformations, dataset_name, dataset_hash, user_metadata
    ):
        # to check that the parent exists on platform - using the hash. See _upload_dataset
        parent = self._assemble_dataset(transformations[0].raw_data_kwargs)

        # setup for generating datasets.
        last = len(transformations) - 1
        upload_queue = list()

        output = dict()
        # iterate over transformations, assembling intermediate datasets
        # TODO address memory issue of keeping all datasets
        for idx, trans in enumerate(transformations):
            output = trans(output)

            # construct the generated dataset from outputs
            dset = self._assemble_dataset(output)
            dset_metadata = copy.deepcopy(user_metadata)
            dset_name = f"{dataset_name}-{trans.func.__name__}"  # TODO improve this.

            # handle the final dataset - check generated = passed in.
            if idx == last:
                if (
                    pd.util.hash_pandas_object(dset).sum() != dataset_hash
                ):  # TODO create or find better exception
                    raise AssertionError(
                        """Generated Dataset does not match the Dataset passed in.
                                     Please check your DatasetTransformation definitions and try again.
                                     Try using less DatasetTransformations if you are having persistent problems"""
                    )
                else:
                    dset_name = dataset_name

            # add data to queue to upload later after final dataset checked
            upload_kwargs = {
                "dataset": copy.deepcopy(dset),  # TODO change keys
                "dataset_name": copy.deepcopy(dset_name),
                "metadata": dset_metadata,
                "parent_hash": pd.util.hash_pandas_object(parent).sum(),
                "transformation": copy.deepcopy(trans),
            }
            # update the parent dataset - these chained transformations only make sense if they are pushing the
            # same dataset through multiple transformations.
            parent = copy.deepcopy(dset)
            upload_queue.append(upload_kwargs)
        return upload_queue

    def upload_training_run_split(
        self,
        model,
        X_train: DataFrame,
        y_train: Union[DataFrame, Series],
        X_test: DataFrame = None,
        y_test: Union[DataFrame, Series] = None,
        X_val: Union[DataFrame, Series] = None,
        y_val: Union[DataFrame, Series] = None,
    ) -> None:
        """
        Takes a model and extracts the necessary data for uploading the training run.

        :param model: An ML Model instance. This should be one of {sklearn.Estimator, xgboost.Booster, lgbm.Boster}.

        :param X_train: Samples of the dataset that the model is trained on

        :param y_train: Labels of the dataset that the model is trained on.

        :param X_test: Samples of the dataset that the model is trained on

        :param y_test: Labels of the dataset that the model is trained on.

        :param X_val: Samples of the dataset that the model is trained on

        :param y_val: Labels of the dataset that the model is trained on.

        :return: None
        """
        train_dataset = self._assemble_dataset({"X": X_train, "y": y_train})
        test_dataset = None
        val_dataset = None
        if X_test is not None and y_test is not None:
            test_dataset = self._assemble_dataset({"X": X_test, "y": y_test})
        if X_val is not None and y_val is not None:
            val_dataset = self._assemble_dataset({"X": X_val, "y": y_val})

        self._storage_q.put(
            {
                "function": "upload_training_run",
                "model": model,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "val_dataset": val_dataset,
                "project": self._project,
            }
        )

        _thread = threading.Thread(target=self._file_processor.writer(self._storage_q))
        _thread.start()
        _sending_thread = threading.Thread(target=self._file_processor.sender())
        _sending_thread.start()

    @staticmethod
    def _assemble_dataset(data: Dict[str, DataFrame]) -> DataFrame:
        if len(data) == 1:
            return next(iter(data.values()))
        elif len(data) == 2:
            # create dataframe from X and y and upload - will have one item in metadata, the output_col
            for key, val in data.items():
                if not (isinstance(val, DataFrame) or isinstance(val, Series)):
                    data[key] = DataFrame(val)
            return pd.concat([x for x in data.values()], axis=1)
        else:
            raise AssertionError(
                "Output doesn't match the requirements. Please review the documentation."
            )

    def _init_project(self, project_name) -> None:
        """
        Initialises the project for the object. If the project does not exist on the server it will be created.

        :param project_name: The name of the project

        :return: None
        """
        self._project = self._get_project(project_name)
        if self._project is None:
            proj_res = self._create_project(project_name=project_name)
            try:
                self._project = proj_res["id"]
            except KeyError:
                # print(f"There was an issue: {proj_res.text}")
                resp = self._transmission.get(
                    url_path="/collection/projects", query_params={"name": project_name}
                )
                resp = handle_response(resp, "There was an issue getting the project")
                self._project = resp.json()[0]["id"]

    def _get_project(self, project_name: str) -> Any:
        """
        Checks if a project exists on the server. If it does not it will return None otherwise the id of the project.

        :param project_name: str The name of the project.

        :return: int | None The id of the project else None.
        """
        project_res = self._transmission.get(
            url_path="/collection/projects",
            query_params={"name": project_name, "organization": self._organization},
        )
        handle_response(project_res, f"There was an issue getting the projects: {project_res.text}")
        if len(project_res.json()) == 0:
            return None
        return project_res.json()[0]["id"]

    def _create_project(self, project_name: str, description: str = "Please add a description.."):
        """
        Creates a new project.

        :param project_name: str The name of the project, must be unique within your Organisation.

        :param description: str Optional The description of the project. This has a default value that can be changed
            at a later date

        :return: Response The response from the server.

        :raises ValueError if the response status is not 201.
        """
        res = asyncio.run(
            self._api.send_json(
                url_path="/collection/projects",
                obj={
                    "name": project_name,
                    "description": description,
                    "organization": self._organization,
                },
                query_params={"organization": self._organization},
                transmission=self._transmission,
                json_response=True,
            )
        )
        return res

    def _set_model(self, model_name: str, framework: ModelManagers) -> int:
        """
        Set the model for this session.
        Checks if it has already been uploaded. If not it will upload it.

        :param model_name: The name for the architecture/algorithm. eg. "GradientBoostedMachine" or "3-layer CNN".

        :return: int The model id.

        :raises: ValueError - if the framework is not one of the supported frameworks or if there is an issue uploading
         the model.
        """
        res = handle_response(
            self._transmission.get(
                url_path="/collection/models",
                query_params={
                    "organization": self._organization,
                    "project": self._project,
                    "name": model_name,
                    "framework": framework.name,
                },
            ),
            msg="There was an issue getting the model list",
        )
        models = res.json()
        if len(models) == 1:
            return models[0]["id"]
        # if we got here that means that the model has not been uploaded yet. So we upload it.
        res = self._upload_model(model_name=model_name, framework=framework)
        try:
            model_pk = res["id"]
        except KeyError:
            resp = handle_response(
                self._transmission.get(
                    url_path="/collection/models",
                    query_params={
                        "organization": self._organization,
                        "project": self._project,
                        "name": model_name,
                        "framework": framework.name,
                    },
                ),
                msg="There was an issue getting the model list",
            )
            model_pk = resp.json()[0]["id"]
        return model_pk

    def _upload_model(self, model_name: str, framework: ModelManagers):
        """

        :param model_name:
        :param framework: instance of seclea_ai.Frameworks
        :return:
        """
        res = asyncio.run(
            self._api.send_json(
                url_path="/collection/models",
                obj={
                    "organization": self._organization,
                    "project": self._project,
                    "name": model_name,
                    "framework": framework.name,
                },
                query_params={"organization": self._organization, "project": self._project},
                transmission=self._transmission,
                json_response=True,
            )
        )
        return res

    def _upload_model_state(
        self,
        model,
        training_run_pk: int,
        sequence_num: int,
        final: bool,
        model_manager: ModelManagers,
    ):
        _thread = threading.Thread(
            target=self._file_processor._save_model_state(
                model,
                training_run_pk,
                sequence_num,
                final,
                model_manager,
            )
        )
        _thread.start()
        _sending_thread = threading.Thread(target=self._file_processor.send_model_state())
        _sending_thread.start()

    def _load_transformations(self, training_run_pk: int):
        """
        Expects a list of code_encoded as set by upload_transformations.
        TODO replace or remove
        """
        res = self._transmission.get(
            url_path="/collection/dataset-transformations",
            query_params={"training_run": training_run_pk},
        )
        res = handle_response(
            res, msg=f"There was an issue loading the transformations: {res.text}"
        )
        transformations = list(map(lambda x: x["code_encoded"], res.json()))
        return list(map(decode_func, transformations))

    @staticmethod
    def _aggregate_dataset(datasets: List[str], index) -> DataFrame:
        """
        Aggregates a list of dataset paths into a single file for upload.
        NOTE the files must be split by row and have the same format otherwise this will fail or cause unexpected format
        issues later.
        :param datasets:
        :return:
        """
        loaded_datasets = [pd.read_csv(dset, index_col=index) for dset in datasets]
        aggregated = pd.concat(loaded_datasets, axis=0)
        return aggregated

    @staticmethod
    def _get_framework(model) -> ModelManagers:
        module = model.__class__.__module__
        # order is important as xgboost and lightgbm contain sklearn compliant packages.
        # TODO check if we can treat them as sklearn but for now we avoid that issue by doing sklearn last.
        if "xgboost" in module:
            return ModelManagers.XGBOOST
        elif "lightgbm" in module:
            return ModelManagers.LIGHTGBM
        elif "sklearn" in module:
            return ModelManagers.SKLEARN
        else:
            return ModelManagers.NOT_IMPORTED
