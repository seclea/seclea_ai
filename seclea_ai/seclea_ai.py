"""
Description for seclea_ai.py
"""
import copy
import inspect
import logging
from pathlib import PurePath, Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas
import pandas as pd
from pandas import DataFrame, Series
from pandas.errors import ParserError
from peewee import SqliteDatabase

from seclea_ai.internal.api.api_interface import Api
from seclea_ai.internal.director import Director
from seclea_ai.internal.exceptions import AuthenticationError
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.core import encode_func

from .lib.seclea_utils.dataset_management.dataset_utils import dataset_hash
from seclea_ai.lib.seclea_utils.model_management.get_model_manager import ModelManagers
from seclea_ai.transformations import DatasetTransformation

logger = logging.getLogger(__name__)


class SecleaAI:
    def __init__(
        self,
        project: str,
        organization: str,
        project_root: str = ".",
        platform_url: str = "https://platform.seclea.com",
        auth_url: str = "https://auth.seclea.com",
        username: str = None,
        password: str = None,
    ):
        """
        Create a SecleaAI object to manage a session. Requires a project name and framework.

        :param project: The name of the project.

        :param organization: The name of the project's organization.

        :param project_root: The path to the root of the project. Default: "."

        :param platform_url: The url of the platform server. Default: "https://platform.seclea.com"

        :param auth_url: The url of the auth server. Default: "https://auth.seclea.com"

        :param username: seclea username

        :param password: seclea password

        :return: SecleaAI object

        :raises: ValueError - if the framework is not supported.

        Example::

            >>> seclea = SecleaAI(project="Test Project", project_root=".")
        """
        self._project = project
        self._organization = organization

        self._settings = {
            "project": project,
            "organization": organization,
            "project_root": project_root,
            "platform_url": platform_url,
            "auth_url": auth_url,
            "cache_dir": PurePath(project_root) / ".seclea" / "cache" / project,
            "offline": False,
        }
        self._db = SqliteDatabase(Path.home() / ".seclea" / "seclea_ai.db", thread_safe=True)
        self._api = Api(
            self._settings, username=username, password=password
        )  # TODO add username and password?
        self._project_id = self._init_project(project=project)
        self._settings["project_id"] = self._project_id
        self._training_run = None
        self._director = Director(settings=self._settings)
        logger.debug("Successfully Initialised SecleaAI class")

    def login(self, username=None, password=None) -> None:
        """
        Override login, this also overwrites the stored credentials in ~/.seclea/config.
        Note. In some circumstances the password will be echoed to stdin. This is not a problem in Jupyter Notebooks
        but may appear in scripting usage.

        :return: None

        Example::

            >>> seclea = SecleaAI(project="Test Project")
            >>> seclea.login()
        """
        for i in range(3):
            try:
                self._api.authenticate(username=username, password=password)
            except AuthenticationError as e:
                print(f"Login attempt {i} failed: {e}")
            else:
                return
        raise AuthenticationError("Failed to login.")

    def complete(self):
        # TODO change to make terminate happen after timeout if specified or something.
        self._director.complete()

    def terminate(self):
        self._director.terminate()

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
            >>> seclea = SecleaAI(project="Test Project")
            >>> dataset = pd.read_csv("/test_folder/dataset_file.csv")
            >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
            >>> seclea.upload_dataset(dataset=dataset, dataset_name="Multifile Dataset", metadata=dataset_metadata)

        Example with file::

            >>> seclea.upload_dataset(dataset="/test_folder/dataset_file.csv", dataset_name="Test Dataset", metadata={})
            >>> seclea = SecleaAI(project="Test Project", organization="Test Organization")

        Assuming the files are all in the /test_folder/dataset directory.
        Example with multiple files::

            >>> files = os.listdir("/test_folder/dataset")
            >>> seclea = SecleaAI(project="Test Project")
            >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
            >>> seclea.upload_dataset(dataset=files, dataset_name="Multifile Dataset", metadata=dataset_metadata)


        """
        # processing the final dataset - make sure it's a DataFrame
        if self._project_id is None:
            raise Exception("You need to create a project before uploading a dataset")

        if isinstance(dataset, List):
            dataset = self._aggregate_dataset(dataset, index=metadata["index"])
        elif isinstance(dataset, str):
            dataset = pd.read_csv(dataset, index_col=metadata["index"])

        # TODO replace with dataset_hash fn
        dataset_id = dataset_hash(dataset, self._project_id)

        if transformations is not None:

            parent = self._assemble_dataset(transformations[0].raw_data_kwargs)

            #####
            # Validate parent exists and get metadata - check how often on portal, maybe remove?
            #####
            parent_dset_id = dataset_hash(parent, self._project_id)
            # check parent exists - check local db if not else error.
            try:
                res = self._api.get_dataset(
                    dataset_id=str(parent_dset_id),
                    organization_id=self._organization,
                    project_id=self._project_id,
                )
            except ValueError:
                # check local db
                self._db.connect()
                parent_record = Record.get_or_none(Record.key == parent_dset_id)
                self._db.close()
                if parent_record is not None:
                    parent_metadata = parent_record.dataset_metadata
                else:
                    raise AssertionError(
                        "Parent Dataset does not exist on the Platform or locally. Please check your arguments and "
                        "that you have uploaded the parent dataset already"
                    )
            else:
                parent_metadata = res.json()["metadata"]
            #####

            upload_queue = self._generate_intermediate_datasets(
                transformations=transformations,
                dataset_name=dataset_name,
                dataset_id=dataset_id,
                user_metadata=metadata,
                parent=parent,
                parent_metadata=parent_metadata,
            )

            # upload all the datasets and transformations.
            for up_kwargs in upload_queue:
                up_kwargs["project"] = self._project_id
                # add to storage and sending queues
                if up_kwargs["entity"] == "dataset":
                    self._director.store_entity(up_kwargs)
                self._director.send_entity(up_kwargs)
            return

        # this only happens if this has no transformations ie. it is a Raw Dataset.

        # validation
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

        # create local db record.
        # TODO make lack of parent more obvious??
        self._db.connect()
        dataset_record = Record.create(
            entity="dataset",
            status=RecordStatus.IN_MEMORY.value,
            key=str(dataset_id),
            dataset_metadata=metadata,
        )
        self._db.close()

        # New arch
        dataset_upload_kwargs = {
            "entity": "dataset",
            "record_id": dataset_record.id,
            "dataset": dataset,
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "metadata": metadata,
            "project": self._project_id,
        }
        # add to storage and sending queues
        self._director.store_entity(dataset_upload_kwargs)
        self._director.send_entity(dataset_upload_kwargs)

    def _generate_intermediate_datasets(
        self, transformations, dataset_name, dataset_id, user_metadata, parent, parent_metadata
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
                split=trans.split if trans is not None else parent_mdata["split"],
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
                    dataset_hash(dset, self._project_id) != dataset_id
                ):  # TODO create or find better exception
                    raise AssertionError(
                        """Generated Dataset does not match the Dataset passed in.
                                     Please check your DatasetTransformation definitions and try again.
                                     Try using less DatasetTransformations if you are having persistent problems"""
                    )
                else:
                    dset_name = dataset_name
            else:
                if dataset_hash(dset, self._project_id) == dataset_hash(
                    parent_dset, self._project_id
                ):
                    raise AssertionError(
                        f"""The transformation {trans.func.__name__} does not change the dataset.
                        Please remove it and try again."""
                    )

            dset_id = dataset_hash(dset, self._project_id)

            # find parent dataset in local db - TODO improve
            self._db.connect()
            parent_record = Record.get_or_none(
                Record.key == dataset_hash(parent_dset, self._project_id)
            )
            parent_dataset_record_id = parent_record.id

            # create local db record.
            dataset_record = Record.create(
                entity="dataset",
                status=RecordStatus.IN_MEMORY.value,
                dependencies=[parent_dataset_record_id],
                key=str(dset_id),
                dataset_metadata=dset_metadata,
            )

            # add data to queue to upload later after final dataset checked
            upload_kwargs = {
                "entity": "dataset",
                "record_id": dataset_record.id,
                "dataset": copy.deepcopy(dset),  # TODO change keys
                "dataset_name": copy.deepcopy(dset_name),
                "dataset_id": dset_id,
                "metadata": dset_metadata,
            }
            # update the parent dataset - these chained transformations only make sense if they are pushing the
            # same dataset through multiple transformations.
            parent_dset = copy.deepcopy(dset)
            parent_mdata = copy.deepcopy(dset_metadata)
            upload_queue.append(upload_kwargs)

            # add dependency to dataset
            trans_record = Record.create(
                entity="transformation",
                status=RecordStatus.IN_MEMORY.value,
                dependencies=[dataset_record.id],
            )
            self._db.close()

            # TODO unpack transformation into kwargs for upload - need to create trans upload func first.
            trans_kwargs = {**trans.data_kwargs, **trans.kwargs}
            transformation_kwargs = {
                "entity": "transformation",
                "record_id": trans_record.id,
                "name": trans.func.__name__,
                "code_raw": inspect.getsource(trans.func),
                "code_encoded": encode_func(trans.func, [], trans_kwargs),
            }
            upload_queue.append(transformation_kwargs)
            # update parent to next dataset in queue
            parent_dataset_record_id = dataset_record.id

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

        self.upload_training_run(model, train_dataset, test_dataset, val_dataset)

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

            >>> seclea = SecleaAI(project="Test Project")
            >>> dataset = pd.read_csv(<dataset_name>)
            >>> model = LogisticRegressionClassifier()
            >>> model.fit(X, y)
            >>> seclea.upload_training_run(
                    model,
                    framework=seclea_ai.Frameworks.SKLEARN,
                    dataset_name="Test Dataset",
                )
        """
        # TODO check if we need this auth check here or if we can do better.
        self._api.authenticate()

        # validate the splits? maybe later when we have proper Dataset class to manage these things.
        dataset_ids = [
            dataset_hash(dataset, self._project_id)
            for dataset in [train_dataset, test_dataset, val_dataset]
            if dataset is not None
        ]

        # Model stuff
        model_name = model.__class__.__name__
        framework = self._get_framework(model)
        # check the model exists upload if not TODO convert to add to queue
        model_type_id = self._set_model(model_name=model_name, framework=framework)

        # check the latest training run TODO extract all this stuff
        training_runs_res = self._api.get_training_runs(
            project_id=self._project_id,
            organization_id=self._organization,
            model=model_type_id,
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

        # search for datasets in local db? Maybe not needed..

        # create record
        self._db.connect()
        training_run_record = Record.create(
            entity="training_run", status=RecordStatus.IN_MEMORY.value
        )

        # sent training run for upload.
        training_run_details = {
            "entity": "training_run",
            "record_id": training_run_record.id,
            "project": self._project_id,
            "training_run_name": training_run_name,
            "model_id": model_type_id,
            "dataset_ids": dataset_ids,
            "params": params,
        }
        self._director.send_entity(training_run_details)

        # create local db record
        model_state_record = Record.create(
            entity="model_state",
            status=RecordStatus.IN_MEMORY.value,
            dependencies=[training_run_record.id],
        )
        self._db.close()

        # send model state for save and upload
        # TODO make a function interface rather than the queue interface. Need a response to confirm it is okay.
        model_state_details = {
            "entity": "model_state",
            "record_id": model_state_record.id,
            "model": model,
            "sequence_num": 0,
            "final": True,
            "model_manager": framework,  # TODO move framework to sender.
        }
        self._director.store_entity(model_state_details)
        self._director.send_entity(model_state_details)

    def _set_model(self, model_name: str, framework: ModelManagers) -> int:
        """
        Set the model for this session.
        Checks if it has already been uploaded. If not it will upload it.zg

        :param model_name: The name for the architecture/algorithm. eg. "GradientBoostedMachine" or "3-layer CNN".

        :return: int The model id.

        :raises: ValueError - if the framework is not one of the supported frameworks or if there is an issue uploading
         the model.
        """
        res = self._api.get_models(
            organization_id=self._organization,
            project_id=self._project_id,
            name=model_name,
            framework=framework.name,
        )
        models = res.json()
        if len(models) == 1:
            return models[0]["id"]
        # if we got here that means that the model has not been uploaded yet. So we upload it.
        res = self._api.upload_model(
            organization_id=self._organization,
            project_id=self._project_id,
            model_name=model_name,
            framework_name=framework.name,
        )
        # TODO find out if this checking is ever needed - ie does it ever not return the created model object?
        try:
            model_id = res.json()["id"]
        except KeyError:
            resp = self._api.get_models(
                organization_id=self._organization,
                project_id=self._project_id,
                name=model_name,
                framework=framework.name,
            )
            model_id = resp.json()[0]["id"]
        return model_id

    @staticmethod
    def _assemble_dataset(
        x: Union[DataFrame, Series], y: Union[DataFrame, Series] = None
    ) -> DataFrame:
        try:
            expected_arg_type = Union[DataFrame, Series]
            if not isinstance(x, expected_arg_type.__args__):
                x = DataFrame(x)

            if y is None:
                return x

            # TODO: review this logic is flawed? typing is specified so val should not be of another type...
            if not isinstance(y, expected_arg_type.__args__):
                y = DataFrame(y)
            return pd.concat([x, y], axis=1)
        except Exception as e:
            raise TypeError(f"Failed to assemble datasets: {type(x), type(y)} with error: {e}")

    def _init_project(self, project) -> int:
        """
        Initialises the project for the object. If the project does not exist on the server it will be created.

        :param project: The name of the project

        :return: None
        """
        project = self._get_project(project)
        if project is None:
            proj_res = self._api.upload_project(
                name=project,
                description="Please add a description...",
                organization_id=self._organization,
            )
            try:
                project = proj_res.json()["id"]
            except KeyError:
                resp = self._api.get_projects(organization_id=self._organization, name=project)
                project = resp.json()[0]["id"]
        return project

    def _get_project(self, project: str) -> Any:
        """
        Checks if a project exists on the server. If it does not it will return None otherwise the id of the project.

        :param project: str The name of the project.

        :return: int | None The id of the project else None.
        """
        project_res = self._api.get_projects(organization_id=self._organization, name=project)
        if len(project_res.json()) == 0:
            return None
        return project_res.json()[0]["id"]

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
            except (ParserError, ValueError):  # Can't convert some
                return "tabular"
            return "time_series"
        return "tabular"

    def dataset_flow(self):
        """
        user specifies reference to data
        user loads some object of type unknown [Dataframe, Serries, Numpy, Tensorflow loaders, pytorch loaders] representing a dataset using specification
        -> We want the user to use our interface to load/manage the above data.
        -> the above needs to stick as close the the original functionality as possible whilst allowing us to control
        """
        ...
