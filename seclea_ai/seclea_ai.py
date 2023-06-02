import copy
import inspect
import logging
import os
import uuid
from uuid import UUID
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from pandas import DataFrame, Series

from .internal.api.api_interface import Api
from .internal.config import read_config
from .internal.dataset_utils import (
    get_dataset_type,
    ensure_required_metadata,
    add_required_metadata,
    aggregate_dataset,
    assemble_dataset,
)
from .internal.exceptions import AuthenticationError, BadRequestError, NotFoundError
from .lib.seclea_utils.core.transformations import encode_func
from .lib.seclea_utils.object_management import Tracked
from .transformations import DatasetTransformation

logging.basicConfig(format="[seclea_ai] [%(levelname)s] %(message)s", level=logging.WARNING)
logger = logging.getLogger("seclea_ai")


class SecleaAI:
    def __init__(
        self,
        project_root: str = ".",
        project_name: str = None,
        organization: str = None,
        platform_url: str = None,
        auth_url: str = None,
        username: str = None,
        password: str = None,
        create_project: bool = False,
    ):
        """
        Create a SecleaAI object to manage a session.
        Can be configured with config files, both globally and per project.
        Init args override project config which overrides global config.
        Project name and organization must be specified somewhere for correct function.

        :param project_root: The root directory of the project. Default: "."

        :param project_name: The name of the project

        :param platform_url: The url of the platform server. Default: "https://platform.seclea.com"

        :param auth_url: The url of the auth server. Default: "https://auth.seclea.com"

        :param username: Username for the seclea platform.

        :param password: Password for the seclea platform.

        :param create_project: Whether to create the project if it does not exist yet
            on the Seclea platform. Default: False.

        :return: SecleaAI object

        :raises: ValueError - if the framework is not supported.

        Example::

            >>> seclea = SecleaAI(project_name="Test Project")
        """
        self._project_name = project_name
        settings_defaults = {
            "project_root": project_root,
            # note this is an exception - must be configured in init or use default.
            "max_storage_space": int(10e9),  # default is 10GB for now.
            "offline": False,
            "cache_dir": (Path(project_root) / ".seclea" / "cache" / project_name).absolute(),
            "platform_url": "https://platform.seclea.com",
            "auth_url": "https://auth.seclea.com",
        }
        # read in the config files - everything can be specified in there except project root.
        global_config = read_config(Path.home() / ".seclea" / "config.yml")
        project_config = read_config(Path(project_root) / ".seclea" / "config.yml")
        # order is important {least_important -> most_important} so default values are first overridden
        self._settings = {**settings_defaults, **global_config, **project_config}

        # apply init args - if specified TODO find a more elegant way to do this.
        if platform_url is not None:
            self._settings["platform_url"] = platform_url
        if auth_url is not None:
            self._settings["auth_url"] = auth_url
        if project_name is not None:
            self._settings["project_name"] = project_name
        if organization is not None:
            self._settings["organization_name"] = organization

        self._validate_settings(["project_name", "organization_name"])

        self._api = Api(settings=self._settings, username=username, password=password)
        self._organization_id = self._init_org(organization)
        self._project_id = self._init_project(
            project_name=project_name, create_project=create_project
        )
        self._training_run = None
        logger.debug("Successfully initialised SecleaAI class")

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
                self._api.authenticate(username=username, password=password)
                success = True
                break
            except AuthenticationError as e:
                print(e)
        if not success:
            raise AuthenticationError("Failed to login.")

    def upload_dataset_split(
        self,
        X: DataFrame,
        y: Union[DataFrame, Series],
        dataset_name: str,
        metadata: Dict,
        transformations: List[DatasetTransformation] = None,
    ) -> None:
        """
        Uploads a dataset.

        :param X: DataFrame The samples of the Dataset.

        :param y: Dataframe The labels of the Dataset

        :param dataset_name: The name of the Dataset

        :param metadata: Any metadata about the Dataset. Required key is:
             "continuous_features"
             To enable bias features include:
                "favorable_outcome"
            Recommended keys:
                "outputs_type": "classification" | "regression"

        :param transformations: A list of DatasetTransformation's.

                        If your Dataset is large try and call this function more often with less DatasetTransformations
                        as the function currently requires (no. DatasetTransformations x Dataset size) memory.

                        See DatasetTransformation for more details.

        :return: None
        """
        # deprecated outcome_name check
        deprecated = metadata.get("outcome_name", None)
        if deprecated is not None:
            raise ValueError("'outcome_name' is deprecated - use 'outputs' instead.")
        dataset = assemble_dataset({"X": X, "y": y})
        # potentially fragile vvv TODO check this vvv
        if isinstance(y, Series):
            metadata["outputs"] = [y.name]
        elif isinstance(y, DataFrame):
            metadata["outputs"] = y.columns.tolist()
        else:
            raise ValueError("y needs to be either Series or DataFrame.")
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
        # deprecated outcome_name check
        deprecated = metadata.get("outcome_name", None)
        if deprecated is not None:
            raise ValueError("outcome_name is deprecated - use outputs instead.")

        # processing the final dataset - make sure it's a DataFrame
        if self._project_id is None:
            raise Exception("You need to create a project before uploading a dataset")

        if isinstance(dataset, List):
            dataset = aggregate_dataset(dataset, index=metadata["index"])
        elif isinstance(dataset, str):
            dataset = pd.read_csv(dataset, index_col=metadata["index"])
        tracked_ds = Tracked(dataset)
        dset_pk = tracked_ds.object_manager.hash_object(tracked_ds)

        if transformations is not None:
            # specific guard against mis specified initial data.
            if not isinstance(list(transformations[0].raw_data_kwargs.values())[0], DataFrame):
                raise ValueError(
                    f"The initial DatasetTransformation data_kwargs must be a DataFrame, found {list(transformations[0].raw_data_kwargs.values())[0]}, of type {type(list(transformations[0].raw_data_kwargs.values())[0])}"
                )

            parent = Tracked(assemble_dataset(transformations[0].raw_data_kwargs))

            #####
            # Validate parent exists and get metadata - can factor out
            #####
            parent_dset_hash = parent.object_manager.hash_object(parent)
            # check parent exists - throw an error if not.
            res = self._api.get_datasets(
                project_id=self._project_id,
                organization_id=self._organization_id,
                hash=parent_dset_hash,
            )
            if len(res) == 0:
                raise AssertionError(
                    "Parent Dataset does not exist on the Platform. Please check your arguments and "
                    "that you have uploaded the parent dataset already"
                )
            parent_metadata = res[0]["metadata"]
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
                if (
                    Tracked(up_kwargs["dataset"]).object_manager.hash_object(up_kwargs["dataset"])
                    == up_kwargs["parent_hash"]
                ):
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
        # set up the defaults - user optional keys
        metadata_defaults_spec = dict(
            continuous_features=[],
            outputs=[],
            outputs_type=None,
            num_samples=len(dataset),
            favourable_outcome=None,
            unfavourable_outcome=None,
            dataset_type=get_dataset_type(dataset),
        )
        metadata = ensure_required_metadata(metadata=metadata, defaults_spec=metadata_defaults_spec)
        # set up automatic values
        try:
            features = (
                dataset.columns
            )  # TODO - drop the outcome name but requires changes on frontend.
        except KeyError:
            # this means outcome was set to None
            features = dataset.columns
        automatic_metadata = dict(
            index=0 if dataset.index.name is None else dataset.index.name,
            outputs_info=tracked_ds.object_manager.get_outputs_info(
                dataset=dataset, outputs=metadata["outputs"], outputs_type=metadata["outputs_type"]
            ),
            split=None,
            features=list(features),
            categorical_features=list(
                set(list(features))
                - set(metadata["continuous_features"]).intersection(set(list(features)))
            ),
            framework=tracked_ds.object_manager.framework,  # needs to be on a Tracked object.
        )
        metadata = add_required_metadata(metadata=metadata, required_spec=automatic_metadata)

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

    @staticmethod
    def _generate_intermediate_datasets(
        transformations, dataset_name, dset_pk, user_metadata, parent, parent_metadata
    ):
        # setup for generating datasets.
        last = len(transformations) - 1
        upload_queue = list()
        parent_mdata = parent_metadata
        parent_dset = parent
        # let user specify outputs_type - defaults to take the parents outputs_type (which has None as default)
        try:
            outputs_type = user_metadata["outputs_type"]
        except KeyError:
            outputs_type = None

        output = dict()
        # iterate over transformations, assembling intermediate datasets
        # TODO address memory issue of keeping all datasets
        for idx, trans in enumerate(transformations):
            output = trans(output)

            # construct the generated dataset from outputs
            dset = assemble_dataset(output)

            dset_metadata = copy.deepcopy(user_metadata)
            # validate and ensure required metadata
            metadata_defaults_spec = dict(
                continuous_features=parent_mdata["continuous_features"],
                outputs=parent_mdata["outputs"],
                outputs_type=parent_mdata["outputs_type"] if outputs_type is None else outputs_type,
                num_samples=len(dset),
                favourable_outcome=parent_mdata["favourable_outcome"],
                unfavourable_outcome=parent_mdata["unfavourable_outcome"],
                dataset_type=get_dataset_type(dset),
            )
            dset_metadata = ensure_required_metadata(
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
                outputs_info=Tracked(dset).object_manager.get_outputs_info(
                    dataset=dset,
                    outputs=dset_metadata["outputs"],
                    outputs_type=dset_metadata["outputs_type"],
                ),
                split=trans.split if trans.split is not None else parent_mdata["split"],
                features=list(features),
                categorical_features=list(
                    set(list(features))
                    - set(dset_metadata["continuous_features"]).intersection(set(list(features)))
                ),
                framework=parent.object_manager.framework,  # needs to be on a Tracked object.
            )

            dset_metadata = add_required_metadata(
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
                    Tracked(dset).object_manager.hash_object(dset) != dset_pk
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
                "dataset": copy.deepcopy(dset),
                "dataset_name": copy.deepcopy(dset_name),
                "metadata": dset_metadata,
                "parent_hash": Tracked(parent_dset).object_manager.hash_object(parent_dset),
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

        :param X_train: Training dataset samples

        :param y_train: Training dataset labels/targets

        :param X_test: Test dataset samples

        :param y_test: Test dataset labels/targets

        :param X_val: Validation dataset samples

        :param y_val: Validation dataset labels/targets

        :return: None
        """
        train_dataset = assemble_dataset({"X": X_train, "y": y_train})
        test_dataset = None
        val_dataset = None
        if X_test is not None and y_test is not None:
            test_dataset = assemble_dataset({"X": X_test, "y": y_test})
        if X_val is not None and y_val is not None:
            val_dataset = assemble_dataset({"X": X_val, "y": y_val})
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
        self._api.authenticate()

        # validate the splits? maybe later when we have proper Dataset class to manage these things.
        dataset_pks = list()
        dataset_metadata = None
        for idx, dataset in enumerate([train_dataset, test_dataset, val_dataset]):
            if dataset is not None:
                try:
                    dataset = self._api.get_datasets(
                        project_id=self._project_id,
                        organization_id=self._organization_id,
                        hash=Tracked(dataset).object_manager.hash_object(dataset),
                    )[0]
                    dataset_metadata = dataset["metadata"]
                    dataset_pks.append(dataset["uuid"])
                except IndexError:
                    # we tried to access [0] of an empty return
                    dset_map = {0: "Train", 1: "Test", 2: "Validation"}
                    raise ValueError(
                        f"The {dset_map[idx]} dataset was not found on the Platform. Please check and try again"
                    )
        model = Tracked(model)

        # check the model exists upload if not
        model_name = str(model.__class__.__name__)
        framework = model.object_manager.framework

        model_type_id = self._set_model(model_name=model_name, framework=framework)

        # check the latest training run
        training_runs = self._api.get_training_runs(
            project_id=self._project_id,
            organization_id=self._organization_id,
            model=model_type_id,
        )

        # Create the training run name
        largest = -1
        for training_run in training_runs:
            num = int(training_run["name"].split(" ")[2])
            if num > largest:
                largest = num
        training_run_name = f"Training Run {largest + 1}"

        # extract params from the model
        params = model.object_manager.get_params(model)

        metadata = {
            "class_name": ".".join([model.__class__.__module__, model.__class__.__name__]),
            "application_type": model.object_manager.get_application_type(
                model, dataset_metadata["outputs_info"]
            ).value,
        }

        # upload training run
        tr_res = self._api.upload_training_run(
            project_id=self._project_id,
            organization_id=self._organization_id,
            datasets=dataset_pks,
            model=model_type_id,
            name=training_run_name,
            params=params,
            metadata=metadata,
        )

        # upload model state. TODO figure out how this fits with multiple model states.
        self._upload_model_state(
            model=model,
            training_run_id=UUID(tr_res["uuid"]),
            dataset_pks=dataset_pks,
            sequence_num=0,
        )

    def _init_org(self, organization_name: str) -> UUID:
        orgs: List[Dict] = self._api.get_organization_by_name(organization_name=organization_name)
        for org in orgs:
            if org["name"] == organization_name:
                return UUID(org["uuid"])
        raise ValueError(
            f"Specified Organization {organization_name} does not exist. Please check and try again."
        )

    def _init_project(self, project_name: str, create_project: bool) -> UUID:
        """
        Initialises the project for the object. If the project does not exist on the server it will be created.

        :param project_name: The name of the project

        :param create_project: Whether to create the project if it doesn't exist on
            the platform

        :return: None
        """
        projects = self._api.get_projects(organization_id=self._organization_id, name=project_name)
        # name and org are unique together so 1 is the max expected
        if len(projects) > 0:
            return UUID(projects[0]["uuid"])
        # this means it is 0 - so it doesn't exist
        if create_project:
            proj_res = self._api.upload_project(
                organization_id=self._organization_id,
                name=project_name,
                description="Please add a description.",
            )
            try:
                return UUID(proj_res["uuid"])
            except KeyError:
                logger.debug(
                    f"There was an issue getting the uuid from the "
                    f"upload_project response: {proj_res.text}"
                )
                resp = self._api.get_projects(
                    organization_id=self._organization_id, name=project_name
                )
                return UUID(resp[0]["uuid"])
        else:
            raise NotFoundError(
                "Project doesn't exist on the Seclea Platform and you "
                "have not set create_project=True. Either "
                "create a new Project on the Platform and try again "
                "or set create_project=True in your SecleaAI init "
                "args."
            )

    def _set_model(self, model_name: str, framework: str) -> UUID:
        """
        Set the model for this session.
        Checks if it has already been uploaded. If not it will upload it.

        :param model_name: The name for the architecture/algorithm. eg. "GradientBoostedMachine" or "3-layer CNN".

        :return: int The model id.

        :raises: ValueError - if the framework is not one of the supported frameworks or if there is an issue uploading
         the model.
        """

        models = self._api.get_models(
            project_id=self._project_id,
            organization_id=self._organization_id,
            name=model_name,
            framework=framework,
        )
        if len(models) == 1:
            return UUID(models[0]["uuid"])
        # model not found on platform - upload it.
        res = self._api.upload_model(
            project_id=self._project_id,
            organization_id=self._organization_id,
            name=model_name,
            framework=framework,
        )
        try:
            model_id = UUID(res["uuid"])
        except KeyError:
            models = self._api.get_models(
                project_id=self._project_id,
                organization_id=self._organization_id,
                name=model_name,
                framework=framework,
            )
            model_id = UUID(models[0]["uuid"])
        return model_id

    def _upload_dataset(
        self,
        dataset: DataFrame,
        dataset_name: str,
        metadata: Dict,
        parent_hash: Union[int, None],
        transformation: Union[DatasetTransformation, None],
    ):
        # upload a dataset - only works for a single transformation.
        os.makedirs(self._settings["cache_dir"], exist_ok=True)

        # TODO refactor to make multithreading safe.
        dataset = Tracked(dataset)
        dataset.object_manager.full_path = self._settings["cache_dir"], uuid.uuid4().__str__()
        dataset_file_path = os.path.join(*dataset.save_tracked(path=self._settings["cache_dir"]))

        dset_hash = Tracked(dataset).object_manager.hash_object(dataset)

        try:
            response = self._api.upload_dataset(
                project_id=self._project_id,
                organization_id=self._organization_id,
                dataset_file_path=dataset_file_path,
                name=dataset_name,
                metadata=metadata,
                hash=dset_hash,
                parent_hash=str(parent_hash) if parent_hash is not None else None,
            )
        # handle duplicate upload with warning only.
        except BadRequestError as e:
            if "fields project, hash must make a unique set" in str(e):
                logger.warning(
                    "You are uploading the same dataset again, "
                    "if this is expected (for example you are re running a script) "
                    "you can ignore this."
                )
            elif "fields project, name must make a unique set" in str(e):
                logger.warning(
                    "You are uploading a dataset with the same name as an "
                    "existing dataset, if this is expected (for example you are "
                    "re-running a script) you can ignore this."
                )
            else:
                raise
        else:
            # upload the transformation
            if transformation is not None:
                # upload transformation
                self._api.upload_transformation(
                    project_id=self._project_id,
                    organization_id=self._organization_id,
                    name=transformation.func.__name__,
                    code_raw=inspect.getsource(transformation.func),
                    code_encoded=encode_func(
                        transformation.func,
                        [],
                        {**transformation.data_kwargs, **transformation.kwargs},
                    ),
                    dataset=response["uuid"],
                )
        finally:
            # tidy up files.
            os.remove(dataset_file_path)

    def _upload_model_state(
        self,
        model: Tracked,
        training_run_id: UUID,
        dataset_pks: List[str],
        sequence_num: int,
    ):
        os.makedirs(
            os.path.join(self._settings["cache_dir"], str(training_run_id)),
            exist_ok=True,
        )
        file_name = f"data-{dataset_pks[0]}-model-{sequence_num}"

        save_path = (
            self._settings["cache_dir"] / str(self._settings["project_name"]) / str(training_run_id)
        )

        model.object_manager.full_path = save_path, file_name

        save_path = os.path.join(*model.save_tracked(path=save_path))

        try:
            self._api.upload_model_state(
                project_id=self._project_id,
                organization_id=self._organization_id,
                model_state_file_path=save_path,
                training_run=training_run_id,
                sequence_num=sequence_num,
            )
        finally:
            # tidy up files.
            os.remove(save_path)

    def _validate_settings(self, required_keys: List[str]) -> None:
        """
        Validates that settings contains non None values for the required keys
        :param required_keys: List[str] list of required keys
        :return: None
        :raises: KeyError: if one of the keys is not present.
        """
        for key in required_keys:
            if self._settings[key] is None:
                raise KeyError(
                    f"Key: {key} must be specified either in creation args or in the config files."
                )
