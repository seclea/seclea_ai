"""
Description for seclea_ai.py
"""
import copy
import inspect
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.errors import ParserError
from requests import Response

from seclea_ai.authentication import AuthenticationService
from seclea_ai.exceptions import AuthenticationError
from seclea_ai.lib.seclea_utils.core import (
    CompressionFactory,
    RequestWrapper,
    decode_func,
    encode_func,
    save_object,
)
from seclea_ai.lib.seclea_utils.model_management.get_model_manager import ModelManagers, serialize
from seclea_ai.transformations import DatasetTransformation
from .lib.seclea_utils.dataset_management.dataset_utils import save_dataset, dataset_hash
from .svc.api.collection.dataset import post_dataset, get_dataset
from .svc.api.collection.model_state import post_model_state


def handle_response(res: Response, expected: int, msg: str) -> Response:
    """
    Handle responses from the server

    :param res: Response The response from the server.

    :param expected: int The expected HTTP status code (ie. 200, 201 etc.)

    :param msg: str The message to include in the Exception that is raised if the response doesn't have the expected
        status code

    :return: Response

    :raises: ValueError - if the response code doesn't match the expected code.

    """
    if not res.status_code == expected:
        raise ValueError(f"Response Status code {res.status_code}, expected:{expected}. \n{msg}")
    return res


class SecleaAI:
    def __init__(
        self,
        project_name: str,
        organization: str,
        platform_url: str = "https://platform.seclea.com",
        auth_url: str = "https://auth.seclea.com",
        username: str = None,
        password: str = None,
    ):
        """
        Create a SecleaAI object to manage a session. Requires a project name and framework.

        :param project_name: The name of the project

        :param platform_url: The url of the platform server. Default: "https://platform.seclea.com"

        :param auth_url: The url of the auth server. Default: "https://auth.seclea.com"

        :param username: seclea username

        :param password: seclea password

        :return: SecleaAI object

        :raises: ValueError - if the framework is not supported.

        Example::

            >>> seclea = SecleaAI(project_name="Test Project")
        """
        self._auth_service = AuthenticationService(RequestWrapper(auth_url))
        self._transmission = RequestWrapper(server_root_url=platform_url)
        self._auth_service.authenticate(self._transmission, username=username, password=password)
        self._project = None
        self._project_name = project_name
        self._organization_name = organization
        self._organization_id = self._init_org(organization)
        self._available_frameworks = {"sklearn", "xgboost", "lightgbm"}
        self._training_run = None
        self._cache_dir = os.path.join(Path.home(), f".seclea/{self._project_name}")
        self._init_project(project_name=project_name)
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

        dset_pk = dataset_hash(dataset, self._project)

        if transformations is not None:

            parent = self._assemble_dataset(transformations[0].raw_data_kwargs)

            #####
            # Validate parent exists and get metadata - can factor out
            #####
            parent_dset_pk = dataset_hash(parent, self._project)
            # check parent exists - throw an error if not.
            res = self._transmission.get(
                url_path="/collection/datasets",
                query_params={
                    "project": self._project,
                    "organization": self._organization_id,
                    "hash": parent_dset_pk,
                },
            )
            if not res.status_code == 200:
                raise AssertionError(
                    "Parent Dataset does not exist on the Platform. Please check your arguments and "
                    "that you have uploaded the parent dataset already"
                )
            parent_metadata = res.json()[0]["metadata"]
            #####

            upload_queue = self._generate_intermediate_datasets(
                transformations=transformations,
                dataset_name=dataset_name,
                dset_pk=dset_pk,
                user_metadata=metadata,
                parent=parent,
                parent_metadata=parent_metadata,
            )

            # check for duplicates
            for up_kwargs in upload_queue:
                if dataset_hash(up_kwargs["dataset"], self._project) == up_kwargs["parent_hash"]:
                    raise AssertionError(
                        f"""The transformation {up_kwargs['transformation'].func.__name__} does not change the dataset.
                        Please remove it and try again."""
                    )
            # upload all the datasets and transformations.
            for up_kwargs in upload_queue:
                # need to get dataset uuid from response to put into transformation.

                self._upload_dataset(**up_kwargs)  # TODO change
            return

        # this only happens if this has no transformations ie. it is a Raw Dataset.
        metadata_defaults_spec = dict(
            continuous_features=[],
            outcome_name=None,
            num_samples=len(dataset),
            favourable_outcome=None,
            unfavourable_outcome=None,
            dataset_type=self._get_dataset_type(dataset),
        )
        try:
            features = (
                dataset.columns
            )  # TODO - drop the outcome name but requires changes on frontend.
        except KeyError:
            # this means outcome was set to None
            features = dataset.columns

        metadata = self._ensure_required_metadata(
            metadata=metadata, defaults_spec=metadata_defaults_spec
        )
        automatic_metadata = dict(
            index=0 if dataset.index.name is None else dataset.index.name,
            split=None,
            features=list(features),
            categorical_features=list(
                set(list(features))
                - set(metadata["continuous_features"]).intersection(set(list(features)))
            ),
        )
        metadata = self._add_required_metadata(metadata=metadata, required_spec=automatic_metadata)

        metadata["categorical_values"] = [
            {col: dataset[col].unique().tolist()} for col in metadata["categorical_features"]
        ]

        self._upload_dataset(
            dataset=dataset,
            dataset_name=dataset_name,
            metadata=metadata,
            parent_hash=None,
            transformation=None,
        )

    def _generate_intermediate_datasets(
        self, transformations, dataset_name, dset_pk, user_metadata, parent, parent_metadata
    ):
        # setup for generating datasets.
        last = len(transformations) - 1
        upload_queue = list()
        parent_mdata = parent_metadata
        parent_dset = parent

        output = dict()
        # iterate over transformations, assembling intermediate datasets
        # TODO address memory issue of keeping all datasets
        for idx, trans in enumerate(transformations):
            output = trans(output)

            # construct the generated dataset from outputs
            dset = self._assemble_dataset(output)

            dset_metadata = copy.deepcopy(user_metadata)
            # validate and ensure required metadata
            metadata_defaults_spec = dict(
                continuous_features=parent_mdata["continuous_features"],
                outcome_name=parent_mdata["outcome_name"],
                num_samples=len(dset),
                favourable_outcome=parent_mdata["favourable_outcome"],
                unfavourable_outcome=parent_mdata["unfavourable_outcome"],
                dataset_type=self._get_dataset_type(dset),
            )
            dset_metadata = self._ensure_required_metadata(
                metadata=dset_metadata, defaults_spec=metadata_defaults_spec
            )
            try:
                features = (
                    dset.columns
                )  # TODO - drop the outcome name but requires changes on frontend.
            except KeyError:
                # this means outcome was set to None
                features = dset.columns

            automatic_metadata = dict(
                index=0 if dset.index.name is None else dset.index.name,
                split=trans.split if trans.split is not None else parent_mdata["split"],
                features=list(features),
                categorical_features=list(
                    set(list(features))
                    - set(dset_metadata["continuous_features"]).intersection(set(list(features)))
                ),
            )

            dset_metadata = self._add_required_metadata(
                metadata=dset_metadata, required_spec=automatic_metadata
            )

            dset_metadata["categorical_values"] = [
                {col: dset[col].unique().tolist()} for col in dset_metadata["categorical_features"]
            ]

            # constraints
            if not set(dset_metadata["continuous_features"]).issubset(
                set(dset_metadata["features"])
            ):
                raise ValueError(
                    "Continuous features must be a subset of features. Please check and try again."
                )

            dset_name = f"{dataset_name}-{trans.func.__name__}"  # TODO improve this.

            # handle the final dataset - check generated = passed in.
            if idx == last:
                if (
                    dataset_hash(dset, self._project) != dset_pk
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
                "parent_hash": dataset_hash(parent_dset, self._project),
                "transformation": copy.deepcopy(trans),
            }
            # update the parent dataset - these chained transformations only make sense if they are pushing the
            # same dataset through multiple transformations.
            parent_dset = copy.deepcopy(dset)
            parent_mdata = copy.deepcopy(dset_metadata)
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
        self.upload_training_run(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            val_dataset=val_dataset,
        )

    def upload_training_run(
        self,
        model,
        train_dataset: DataFrame,
        test_dataset: DataFrame = None,
        val_dataset: DataFrame = None,
    ) -> None:
        """
        Takes a model and extracts the necessary data for uploading the training run.

        :param model: An ML Model instance. This should be one of {sklearn.Estimator, xgboost.Booster, lgbm.Boster}.

        :param train_dataset: DataFrame The Dataset that the model is trained on.

        :param test_dataset: DataFrame The Dataset that the model is trained on.

        :param val_dataset: DataFrame The Dataset that the model is trained on.

        :return: None

        Example::

            >>> seclea = SecleaAI(project_name="Test Project")
            >>> dataset = pd.read_csv(<dataset_name>)
            >>> model = LogisticRegressionClassifier()
            >>> model.fit(X, y)
            >>> seclea.upload_training_run(
                    model,
                    framework=seclea_ai.Frameworks.SKLEARN,
                    dataset_name="Test Dataset",
                )
        """
        self._auth_service.authenticate(self._transmission)

        # validate the splits? maybe later when we have proper Dataset class to manage these things.
        dataset_pks = [
            get_dataset(
                self._transmission,
                self._project,
                self._organization_id,
                hash=dataset_hash(dataset, self._project),
            ).json()[0]["uuid"]
            for dataset in [train_dataset, test_dataset, val_dataset]
            if dataset is not None
        ]

        model_name = model.__class__.__name__

        framework = self._get_framework(model)

        # check the model exists upload if not
        model_type_pk = self._set_model(model_name=model_name, framework=framework)

        # check the latest training run
        training_runs_res = self._transmission.get(
            "/collection/training-runs",
            query_params={
                "project": self._project,
                "model": model_type_pk,
                "organization": self._organization_id,
            },
        )
        training_runs = training_runs_res.json()

        # Create the training run name
        largest = -1
        for training_run in training_runs:
            num = int(training_run["name"].split(" ")[2])
            if num > largest:
                largest = num
        training_run_name = f"Training Run {largest + 1}"

        # extract params from the model
        params = framework.value.get_params(model)

        # upload training run
        tr_res = self._upload_training_run(
            training_run_name=training_run_name,
            model_pk=model_type_pk,
            dataset_pks=dataset_pks,
            params=params,
        )
        # if the upload was successful, add the new training_run to the list to keep the names updated.
        self._training_run = tr_res.json()["uuid"]

        # upload model state. TODO figure out how this fits with multiple model states.
        self._upload_model_state(
            model=model,
            training_run_pk=self._training_run,
            dataset_pks=dataset_pks,
            sequence_num=0,
            final=True,
            model_manager=framework,
        )

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

    def _init_org(self, organization_name: str) -> str:
        orgs_res = self._transmission.get(url_path="/organization/")
        orgs_res = handle_response(
            orgs_res, 200, f"There was an issue getting the project: {orgs_res.reason}"
        )
        orgs = orgs_res.json()
        for org in orgs:
            if org["name"] == organization_name:
                return org["uuid"]
        raise ValueError(
            f"Specified Organization {self._organization_name} does not exist. Please check and try again."
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
                self._project = proj_res.json()["uuid"]
            except KeyError:
                print(f"There was an issue: {proj_res.text}")
                resp = self._transmission.get(
                    url_path="/collection/projects", query_params={"name": project_name}
                )
                resp = handle_response(
                    resp, 200, f"There was an issue getting the project: {resp.reason}"
                )
                self._project = resp.json()[0]["uuid"]

    def _get_project(self, project_name: str) -> Any:
        """
        Checks if a project exists on the server. If it does not it will return None otherwise the id of the project.

        :param project_name: str The name of the project.

        :return: int | None The id of the project else None.
        """
        project_res = self._transmission.get(
            url_path="/collection/projects",
            query_params={"name": project_name, "organization": self._organization_id},
        )
        handle_response(
            project_res, 200, f"There was an issue getting the projects: {project_res.reason}"
        )
        if len(project_res.json()) == 0:
            return None
        return project_res.json()[0]["uuid"]

    def _create_project(self, project_name: str, description: str = "Please add a description.."):
        """
        Creates a new project.

        :param project_name: str The name of the project, must be unique within your Organisation.

        :param description: str Optional The description of the project. This has a default value that can be changed
            at a later date

        :return: Response The response from the server.

        :raises ValueError if the response status is not 201.
        """
        res = self._transmission.send_json(
            url_path="/collection/projects",
            obj={
                "name": project_name,
                "description": description,
                "organization": self._organization_id,
            },
            query_params={"organization": self._organization_id},
        )
        return handle_response(
            res, expected=201, msg=f"There was an issue creating the project: {res.reason}"
        )

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
                    "organization": self._organization_id,
                    "project": self._project,
                    "name": model_name,
                    "framework": framework.name,
                },
            ),
            expected=200,
            msg="There was an issue getting the model list",
        )
        models = res.json()
        if len(models) == 1:
            return models[0]["uuid"]
        # if we got here that means that the model has not been uploaded yet. So we upload it.
        res = self._upload_model(model_name=model_name, framework=framework)
        try:
            model_pk = res.json()["uuid"]
        except KeyError:
            resp = handle_response(
                self._transmission.get(
                    url_path="/collection/models",
                    query_params={
                        "organization": self._organization_id,
                        "project": self._project,
                        "name": model_name,
                        "framework": framework.name,
                    },
                ),
                expected=200,
                msg="There was an issue getting the model list",
            )
            model_pk = resp.json()[0]["uuid"]
        return model_pk

    @staticmethod
    def _ensure_required_metadata(metadata: Dict, defaults_spec: Dict) -> Dict:
        """
        Ensures that required metadata that can be specified by the user are filled.
        @param metadata: The metadata dict
        @param defaults_spec:
        @return: metadata
        """
        for required_key, default in defaults_spec.items():
            try:
                if metadata[required_key] is None:
                    metadata[required_key] = default
            except KeyError:
                metadata[required_key] = default
        return metadata

    @staticmethod
    def _add_required_metadata(metadata: Dict, required_spec: Dict) -> Dict:
        """
        Adds required - non user specified fields to the metadata
        @param metadata: The metadata dict
        @param required_spec:
        @return: metadata
        """
        for required_key, default in required_spec.items():
            metadata[required_key] = default
        return metadata

    def _upload_dataset(
        self,
        dataset: DataFrame,
        dataset_name: str,
        metadata: Dict,
        parent_hash: Union[int, None],
        transformation: Union[DatasetTransformation, None],
    ):
        # upload a dataset - only works for a single transformation.
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # TODO refactor to make multithreading safe.
        dataset_file_path = save_dataset(
            dataset, dataset_name, os.path.join(self._cache_dir, uuid.uuid4().__str__())
        )
        dset_pk = dataset_hash(dataset, self._project)

        response = post_dataset(
            transmission=self._transmission,
            dataset_file_path=dataset_file_path,
            project_pk=self._project,
            organization_pk=self._organization_id,
            name=dataset_name,
            metadata=metadata,
            dataset_pk=dset_pk,
            parent_dataset_hash=str(parent_hash) if parent_hash is not None else None,
        )
        handle_response(
            response, 201, f"There was some issue uploading the dataset: {response.reason}"
        )
        # tidy up files.
        os.remove(dataset_file_path)

        # upload the transformations
        if response.status_code == 201:
            if transformation is not None:
                # upload transformations.
                self._upload_transformation(
                    transformation=transformation,
                    dataset_pk=response.json()["uuid"],
                )

    def _upload_model(self, model_name: str, framework: ModelManagers):
        """

        :param model_name:
        :param framework: instance of seclea_ai.Frameworks
        :return:
        """
        res = self._transmission.send_json(
            url_path="/collection/models",
            obj={
                "organization": self._organization_id,
                "project": self._project,
                "name": model_name,
                "framework": framework.name,
            },
            query_params={"organization": self._organization_id, "project": self._project},
        )
        return handle_response(
            res, expected=201, msg=f"There was an issue uploading the model: {res.reason}"
        )

    def _upload_training_run(
        self, training_run_name: str, model_pk: int, dataset_pks: List[str], params: Dict
    ):
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
                "organization": self._organization_id,
                "project": self._project,
                "datasets": dataset_pks,
                "model": model_pk,
                "name": training_run_name,
                "params": params,
            },
            query_params={"organization": self._organization_id, "project": self._project},
        )
        return handle_response(
            res, expected=201, msg=f"There was an issue uploading the training run: {res.reason}"
        )

    def _upload_model_state(
        self,
        model,
        training_run_pk: int,
        dataset_pks: List[str],
        sequence_num: int,
        final: bool,
        model_manager: ModelManagers,
    ):
        os.makedirs(
            os.path.join(self._cache_dir, str(training_run_pk)),
            exist_ok=True,
        )
        model_data = serialize(model, model_manager)
        file_name = f"data-{dataset_pks[0]}-model-{sequence_num}"
        save_path = os.path.join(Path.home(), f".seclea/{self._project_name}/{training_run_pk}")
        save_path = save_object(
            model_data, file_name, save_path, compression=CompressionFactory.ZSTD
        )

        res = post_model_state(
            self._transmission,
            save_path,
            self._organization_id,
            self._project,
            str(training_run_pk),
            sequence_num,
            final,
        )

        # tidy up files.
        os.remove(save_path)

        res = handle_response(
            res, expected=201, msg=f"There was an issue uploading a model state: {res.reason}"
        )
        return res

    def _upload_transformation(self, transformation: DatasetTransformation, dataset_pk):
        idx = 0
        trans_kwargs = {**transformation.data_kwargs, **transformation.kwargs}
        data = {
            "name": transformation.func.__name__,
            "code_raw": inspect.getsource(transformation.func),
            "code_encoded": encode_func(transformation.func, [], trans_kwargs),
            "order": idx,
            "dataset": dataset_pk,
        }
        res = self._transmission.send_json(
            url_path="/collection/dataset-transformations",
            obj=data,
            query_params={"organization": self._organization_id, "project": self._project},
        )
        res = handle_response(
            res,
            expected=201,
            msg=f"There was an issue uploading the transformations on transformation {idx} with name {transformation.func.__name__}: {res.reason}",
        )
        return res

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
            res, expected=200, msg=f"There was an issue loading the transformations: {res.reason}"
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
        # TODO improve to exclude other keras backends...
        elif "tensorflow" in module or "keras" in module:
            return ModelManagers.TENSORFLOW
        elif "sklearn" in module:
            return ModelManagers.SKLEARN
        else:
            return ModelManagers.NOT_IMPORTED

    @staticmethod
    def _get_dataset_type(dataset: DataFrame) -> str:
        if not np.issubdtype(dataset.index.dtype, np.integer):
            try:
                pd.to_datetime(dataset.index.values)
            except (ParserError, ValueError):  # Can't cnvrt some
                return "tabular"
            return "time_series"
        return "tabular"
