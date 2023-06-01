"""
Description for seclea_ai.py
"""
import copy
import inspect
import logging
import traceback
import uuid
from pathlib import Path
from pathlib import PurePath
from typing import Dict, List, Union, Tuple

import pandas as pd
from pandas import DataFrame, Series
from peewee import SqliteDatabase

from .internal.persistence.models import Project, Dataset, TrainingRun, Model, ModelState
from .internal.schemas import (
    DatasetSchema,
    ProjectSchema,
    ProjectDBSchema,
    DatasetTransformationSchema,
    TrainingRunSchema,
    ModelStateSchema,
)
from .internal.persistence import models
from .internal.api.api_interface import Api
from .internal.director import Director
from .internal.exceptions import AuthenticationError, BadRequestError
from .internal.persistence.record import Record, RecordStatus
from .lib.seclea_utils.core.transformations import encode_func
from .dataset_utils import (
    get_dataset_type,
    ensure_required_metadata,
    add_required_metadata,
    aggregate_dataset,
    assemble_dataset,
)
from .transformations import DatasetTransformation
from .internal.config import read_config
from .lib.seclea_utils.object_management import Tracked

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
        clean_up: bool = False,
    ):
        """
        Create a SecleaAI object to manage a session. Requires a project name and framework.

        :param project_root: The path to the root of the project. Default: "."

        :param project_name: The name of the project.

        :param organization: The name of the project's organization.

        :param platform_url: The url of the platform server. Default: "https://platform.seclea.com"

        :param auth_url: The url of the auth server. Default: "https://auth.seclea.com"

        :param username: seclea username

        :param password: seclea password

        :return: SecleaAI object

        Example::

            >>> seclea = SecleaAI(project_name="Test Project", organization="Test Org", project_root=".")
        """
        self._project_name = project_name
        settings_defaults = {
            "project_root": project_root,  # note this is an exception - must be configured in init or use default.
            "max_storage_space": int(10e9),  # default is 10GB for now.
            "offline": False,
            "cache_dir": (Path(project_root) / ".seclea" / "cache" / project_name).absolute(),
            "platform_url": "https://platform.seclea.com",
            "auth_url": "https://auth.seclea.com",
            "db": dict(
                database=Path.home() / ".seclea" / "seclea_ai.db",
                thread_safe=True,
                pragmas={"journal_mode": "wal"},
            ),
        }
        # read in the config files - everything can be specified in there except project root.
        global_config = read_config(Path.home() / ".seclea" / "config.yml")
        project_config = read_config(PurePath(project_root) / ".seclea" / "config.yml")
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

        self._db = SqliteDatabase(**self._settings.get("db", dict()))
        self._api = Api(
            self._settings, username=username, password=password
        )  # TODO add username and password?
        self._organization_name = organization
        self._organization_id = self._init_org(organization)
        self._project, self._project_id = self._init_project(project_name=project_name)
        self._settings["project_id"] = self._project_id
        self._settings["organization_id"] = self._organization_id
        self._available_frameworks = {"sklearn", "xgboost", "lightgbm"}
        self._training_run = None
        self._director = Director(settings=self._settings, api=self._api, db=self._db)
        logger.debug("Successfully Initialised SecleaAI class")
        if clean_up:
            logger.info("Trying to clean up previous session")
            self._director.try_cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.complete()
        return False

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

    def complete(self):
        # TODO change to make terminate happen after timeout if specified or something.
        self._director.complete()

    def terminate(self):
        self._director.terminate()

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

        if self._project_id is None:
            raise Exception("You need to create a project before uploading a dataset")

        # processing the final dataset - make sure it's a DataFrame
        if isinstance(dataset, List):
            dataset = aggregate_dataset(dataset, index=metadata["index"])
        elif isinstance(dataset, str):
            dataset = pd.read_csv(dataset, index_col=metadata["index"])

        tracked_ds = Tracked(dataset)
        dset_hash = tracked_ds.object_manager.hash_object_with_project(
            tracked_ds, str(self._project_id)
        )

        if transformations is not None:
            # specific guard against mis specified initial data.
            if not isinstance(list(transformations[0].raw_data_kwargs.values())[0], DataFrame):
                raise ValueError(
                    f"The initial DatasetTransformation data_kwargs must be a "
                    f"DataFrame, found "
                    f"{list(transformations[0].raw_data_kwargs.values())[0]}, of type "
                    f"{type(list(transformations[0].raw_data_kwargs.values())[0])}"
                )

            parent = Tracked(assemble_dataset(transformations[0].raw_data_kwargs))

            #####
            # Validate parent exists and get metadata - check how often on portal, maybe remove?
            #####
            parent_dset_hash = parent.object_manager.hash_object_with_project(
                parent, str(self._project_id)
            )

            # check parent exists - check local db if not else error.
            # TODO remove request - only check local db once syncing resolved.
            res = self._api.get_datasets(
                organization_id=self._organization_id,
                project_id=self._project_id,
                hash=parent_dset_hash,
            )
            if len(res) == 0:
                # check local db
                with self._db.atomic():
                    parent_dataset = Dataset.get_or_none(Dataset.hash == parent_dset_hash)
                if parent_dataset is not None:
                    parent_metadata = parent_dataset.metadata
                else:
                    raise AssertionError(
                        "Parent Dataset does not exist on the Platform or locally. Please check your arguments and "
                        "that you have uploaded the parent dataset already"
                    )
            else:
                parent_metadata = res[0]["metadata"]
            #####

            upload_queue = self._generate_intermediate_datasets(
                transformations=transformations,
                dataset_name=dataset_name,
                final_dataset_hash=dset_hash,
                user_metadata=metadata,
                parent=parent,
                parent_metadata=parent_metadata,
            )

            # upload all the datasets and transformations.
            for up_kwargs in upload_queue:
                if up_kwargs["type"] == DatasetSchema:
                    self._director.store_entity(up_kwargs)
                self._director.send_entity(up_kwargs)
            return

        # this only happens if this has no transformations ie. it is a Raw Dataset.
        # set up the defaults - user optional keys

        # validation
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
        features = list(dataset.columns)
        automatic_metadata = dict(
            index=0 if dataset.index.name is None else dataset.index.name,
            outputs_info=tracked_ds.object_manager.get_outputs_info(
                dataset=dataset, outputs=metadata["outputs"], outputs_type=metadata["outputs_type"]
            ),
            split=None,
            features=features,
            categorical_features=list(
                set(features) - set(metadata["continuous_features"]).intersection(set(features))
            ),
            framework=tracked_ds.object_manager.framework,  # needs to be on a Tracked object.
        )
        metadata = add_required_metadata(metadata=metadata, required_spec=automatic_metadata)

        metadata["categorical_values"] = [
            {col: dataset[col].unique().tolist()} for col in metadata["categorical_features"]
        ]

        # create local db record.
        # TODO optimize db accesses
        with self._db.atomic():
            # TODO check for dataset with this hash
            dataset_record = Record.create(
                status=RecordStatus.IN_MEMORY,
            )
            # TODO this will throw error on duplicate name, hash etc. need to catch
            #  and warn user - probably just use existing dset though.
            dataset_entity = models.Dataset.create(
                uuid=uuid.uuid4(),
                name=dataset_name,
                hash=dset_hash,
                metadata=metadata,
                project=Project.get_by_id(self._project.id),
                record=dataset_record,
            )
            print(f"dataset project type: {type(dataset_entity.project)}")

        # new new arch
        dataset_entity_model = DatasetSchema.from_orm(dataset_entity)
        dataset_entity_model.dataset = copy.deepcopy(dataset)
        dataset_upload_kwargs = {"type": DatasetSchema, "entity": dataset_entity_model.dict()}

        # add to storage and sending queues
        self._director.store_entity(dataset_upload_kwargs)
        self._director.send_entity(dataset_upload_kwargs)

    def _generate_intermediate_datasets(
        self,
        transformations,
        dataset_name,
        final_dataset_hash,
        user_metadata,
        parent,
        parent_metadata,
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
        for idx, transformation in enumerate(transformations):
            output = transformation(output)

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

            features = list(dset.columns)

            automatic_metadata = dict(
                index=0 if dset.index.name is None else dset.index.name,
                outputs_info=Tracked(dset).object_manager.get_outputs_info(
                    dataset=dset,
                    outputs=dset_metadata["outputs"],
                    outputs_type=dset_metadata["outputs_type"],
                ),
                split=transformation.split
                if transformation.split is not None
                else parent_mdata["split"],
                features=features,
                categorical_features=list(
                    set(features)
                    - set(dset_metadata["continuous_features"]).intersection(set(features))
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

            dset_name = f"{dataset_name}-{transformation.func.__name__}"  # TODO improve this.
            dset_hash = Tracked(dset).object_manager.hash_object_with_project(
                dset, str(self._project_id)
            )

            # handle the final dataset - check generated = passed in.
            if idx == last:
                if dset_hash != final_dataset_hash:
                    # TODO create or find better exception
                    raise AssertionError(
                        "Generated Dataset does not match the Dataset passed in.\n"
                        "Please check your DatasetTransformation definitions and "
                        "try again. Try using less DatasetTransformations if you "
                        "are having persistent problems"
                    )
                else:
                    dset_name = dataset_name

            parent_hash = Tracked(parent_dset).object_manager.hash_object_with_project(
                parent_dset, str(self._project_id)
            )
            if dset_hash == parent_hash:
                raise AssertionError(
                    f"The transformation {transformation.func.__name__} does not "
                    f"change the dataset.Please remove it and try again."
                )

            # here we will fetch the parent Dataset, create a Record and link them into
            # a new Dataset object. That will then be serialised and added to the queue
            # TODO optimize db accesses
            with self._db.atomic():

                # create local db record.
                dataset_record = Record.create(
                    status=RecordStatus.IN_MEMORY,
                )
                dataset_entity = models.Dataset.create(
                    uuid=uuid.uuid4(),
                    name=dset_name,
                    hash=dset_hash,
                    metadata=dset_metadata,
                    project=Project.get_by_id(self._project.id),
                    parent=Dataset.get_or_none(Dataset.hash == parent_hash),
                    record=dataset_record,
                )

            # new new arch
            dataset_entity_model = DatasetSchema.from_orm(dataset_entity)
            # add dataset reference to pass for storage.
            dataset_entity_model.dataset = copy.deepcopy(dset)
            # TODO rethink this. Would like it to be consistent for all types.
            upload_kwargs = {"type": DatasetSchema, "entity": dataset_entity_model.dict()}

            # update the parent dataset - these chained transformations only make sense if they are pushing the
            # same dataset through multiple transformations.
            parent_dset = copy.deepcopy(dset)
            parent_mdata = copy.deepcopy(dset_metadata)
            upload_queue.append(upload_kwargs)

            # transformation
            transformation_kwargs = {**transformation.data_kwargs, **transformation.kwargs}
            with self._db.atomic():
                # add dependency to dataset
                transformation_record = Record.create(
                    status=RecordStatus.IN_MEMORY,
                )
                transformation_entity = models.DatasetTransformation.create(
                    uuid=uuid.uuid4(),
                    name=transformation.func.__name__,
                    code_raw=inspect.getsource(transformation.func),
                    code_encoded=encode_func(transformation.func, [], transformation_kwargs),
                    dataset=dataset_entity,
                    record=transformation_record,
                )
            # validate and get schema
            transformation_entity_model = DatasetTransformationSchema.from_orm(
                transformation_entity
            )
            # serialize
            transformation_upload_kwargs = {
                "type": DatasetTransformationSchema,
                "entity": transformation_entity_model.dict(),
            }

            # TODO unpack transformation into kwargs for upload - need to create trans upload func first.
            # transformation_upload_kwargs = {
            #     "entity": RecordEntity.DATASET_TRANSFORMATION,
            #     "record_id": transformation_record.id,
            #     "name": transformation.func.__name__,
            #     "code_raw": inspect.getsource(transformation.func),
            #     "code_encoded": encode_func(transformation.func, [], transformation_kwargs),
            # }
            upload_queue.append(transformation_upload_kwargs)

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

        :param model: An ML Model instance. This should be one of {sklearn.Estimator, xgboost.Booster, lgbm.Booster}.

        :param X_train: Samples of the models training dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param y_train: Labels of the models training dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param X_test: Samples of the models test dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param y_test: Labels of the models test dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param X_val: Samples of the models validation dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param y_val: Labels of the models validation dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

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

        :param train_dataset: DataFrame The models training dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param test_dataset: DataFrame The models test dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

        :param val_dataset: DataFrame The models validation dataset. Must be already
            uploaded using `upload_dataset` or `upload_dataset_split`

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
        # TODO check if we need this auth check here or if we can do better.
        self._api.authenticate()

        # validate the splits? maybe later when we have proper Dataset class to manage these things.
        datasets = list()
        dataset_metadata = None
        for idx, dataset in enumerate([train_dataset, test_dataset, val_dataset]):
            if dataset is not None:
                # TODO add db connection management - here auto for now.
                dset_hash = Tracked(dataset).object_manager.hash_object_with_project(
                    dataset, str(self._project_id)
                )
                dataset_entity = Dataset.get_or_none(Dataset.hash == dset_hash)
                if dataset_entity is None:
                    # we tried to access [0] of an empty return
                    dset_map = {0: "Train", 1: "Test", 2: "Validation"}
                    raise ValueError(
                        f"The {dset_map[idx]} dataset was not found on the Platform. "
                        f"Please check and try again"
                    )
                else:
                    dataset_metadata = dataset_entity.metadata
                    datasets.append(dataset_entity)

        # Model stuff
        model_name = model.__class__.__name__
        model = Tracked(model)
        framework = model.object_manager.framework  # needs to be on Tracked

        # check the model exists upload if not
        model_type = self._set_model_type(model_name=model_name, framework=framework)

        # check the latest training run
        # TODO rethink training run naming.
        training_runs = (
            TrainingRun.select()
            .where(TrainingRun.project == self._project.id)
            .where(TrainingRun.model == model_type)
        )

        # Create the training run name
        largest = -1
        for training_run in training_runs:
            num = int(training_run.name.split(" ")[2])
            if num > largest:
                largest = num
        training_run_name = f"Training Run {largest + 1}"

        # extract params from the model
        model = Tracked(model)
        params = model.object_manager.get_params(model)

        metadata = {
            "class_name": ".".join([model.__class__.__module__, model.__class__.__name__]),
            "application_type": model.object_manager.get_application_type(
                model, dataset_metadata["outputs_info"]
            ).value,
        }

        # create record
        with self._db.atomic():
            training_run_record = Record.create(
                status=RecordStatus.IN_MEMORY,
            )
            training_run_entity = TrainingRun.create(
                uuid=uuid.uuid4(),
                name=training_run_name,
                metadata=metadata,
                params=params,
                project=self._project.id,
                model=model_type,
                record=training_run_record,
            )
            training_run_entity.datasets.add(datasets)

        training_run_entity_model = TrainingRunSchema.from_orm(training_run_entity)
        training_run_upload_kwargs = {
            "type": TrainingRunSchema,
            "entity": training_run_entity_model.dict(),
        }

        self._director.send_entity(training_run_upload_kwargs)

        with self._db.atomic():
            # create local db record
            model_state_record = Record.create(
                status=RecordStatus.IN_MEMORY,
            )
            # TODO add final to model - figure out migration?
            model_state_entity = ModelState.create(
                uuid=uuid.uuid4(),
                sequence_num=0,
                training_run=training_run_entity,
                record=model_state_record,
            )
        model_state_entity_model = ModelStateSchema.from_orm(model_state_entity)
        # add the model reference for passing to writer.
        model_state_entity_model.state = model
        # TODO think - do we need to send the schema or just uuid - fetch from db?
        #  or we don't store anything in db except the paths and just pass everything
        #  else around in the schema/dict serialization of it. (that is faster than
        #  always fetching from the db)
        model_state_upload_kwargs = {
            "type": ModelStateSchema,
            "entity": model_state_entity_model.dict(),
        }

        self._director.store_entity(model_state_upload_kwargs)
        self._director.send_entity(model_state_upload_kwargs)

    def _set_model_type(self, model_name: str, framework: str) -> Model:
        with self._db.atomic():
            model, created = Model.get_or_create(
                name=model_name,
                framework=framework,
            )
        if created:
            try:
                self._api.upload_model(
                    organization_id=self._organization_id,
                    project_id=self._project_id,
                    model_name=model.name,
                    framework_name=model.framework,
                    model_id=model.uuid,
                )
            except BadRequestError as e:
                # need to check content - if it's duplicate we need to get the remote id for use in other reqs
                logger.debug(e)
                if "already exists" in str(e):
                    logger.warning(
                        f"Entity already exists, updating local Model, " f"id: {model.uuid}"
                    )
                    models = self._api.get_models(
                        organization_id=self._organization_id,
                        project_id=self._project_id,
                        name=model.name,
                        framework=model.framework,
                    )
                    # update uuid with remote uuid
                    model.uuid = models[0]["uuid"]
                    model.save()
        return model

    def _init_org(self, organization_name: str) -> uuid.UUID:
        org_res = self._api.get_organization(organization_name=organization_name)
        if len(org_res) == 0:
            raise ValueError(
                f"Specified Organization {self._organization_name} does not exist. Please check and try again."
            )
        else:
            return uuid.UUID(org_res[0]["uuid"])

    def _init_project(self, project_name) -> Tuple[ProjectDBSchema, uuid.UUID]:
        """
        Initialises the project for the object. If the project does not exist on the server it will be created.

        :param project_name: The name of the project

        :return: None
        """
        try:
            project_json = self._api.get_projects(
                organization_id=self._organization_id,
                name=project_name,
            )
            project = ProjectSchema.parse_obj(project_json[0])
        # errors if list is empty
        except IndexError:
            proj_res = self._api.upload_project(
                name=project_name,
                description="Please add a description...",
                organization_id=self._organization_id,
            )
            try:
                project = ProjectSchema.parse_obj(proj_res)
            except KeyError:
                # we want to know when this happens as this is unusual condition.
                traceback.print_exc()
                resp = self._api.get_projects(
                    organization_id=self._organization_id, name=project_name
                )
                project = ProjectSchema.parse_obj(resp[0])
        # update db.
        with self._db.atomic():
            project, _ = Project.get_or_create(**project.dict())
            project = ProjectDBSchema.from_orm(project)
        return project, project.uuid

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
