from __future__ import annotations

import abc
import os
from abc import abstractmethod
from pathlib import Path
from typing import List

from peewee import SqliteDatabase

from seclea_ai.internal.api.api_interface import Api
from seclea_ai.internal.director import Director
from seclea_ai.internal.exceptions import NotFoundError
from seclea_ai.lib.seclea_utils.object_management.object_manager import ToFileMixin, MetadataMixin


class SerializerMixin(metaclass=abc.ABCMeta):
    @staticmethod
    def get_serialized(obj, meta_list: List[SerializerMixin.__class__]) -> dict:
        ser_list = [meta.serialize(obj) for meta in meta_list]
        result = dict()
        for s in ser_list:
            result.update(s)
        return result

    def deserialize(self, obj: dict):
        for key, val in obj:
            setattr(self, key, val)

    @abstractmethod
    def serialize(self):
        raise NotImplementedError


class APIMixin:
    _api: Api

    @property
    def api(self):
        return self._api

    @api.setter
    def api(self, val: Api):
        self._api = val


class DescriptionMixin(SerializerMixin):
    _description: str

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, val):
        self._description = val

    def serialize(self):
        return {'description': self.description}


class NameMixin(SerializerMixin):
    _name: str

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    def serialize(self):
        return {'name': self.name}


class IDMixin(SerializerMixin):
    _uuid: str

    @property
    def uuid(self):
        return self._uuid

    @uuid.setter
    def uuid(self, val):
        pass

    def serialize(self):
        return {'uuid': self.uuid}


class UsernameMixin(SerializerMixin):
    _username: str

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, val):
        pass

    def serialize(self):
        return {'username': self.username}


class EmailMixin(SerializerMixin):
    _email: str

    @property
    def email(self):
        return self._email

    @email.setter
    def email(self, val):
        pass

    def serialize(self):
        return {'email': self.email}


class PasswordMixin(SerializerMixin):
    _password: str

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, val):
        pass

    def serialize(self):
        return {'password': self.password}


class OrganizationMixin(IDMixin, NameMixin, SerializerMixin):
    def __init__(self, name: str):
        self.name = name


class ProjectMixin(IDMixin, NameMixin, DescriptionMixin, SerializerMixin):
    organization: OrganizationMixin

    def deserialize(self, obj: dict):
        # remove organization as we don't want to overwrite
        self.organization.uuid = obj.pop('organization')
        SerializerMixin.deserialize(self, obj)

    def serialize(self):
        return {**self.get_serialized(self, [IDMixin, NameMixin]), "organization": self.organization.uuid}


class UserMixin(IDMixin, UsernameMixin, PasswordMixin, EmailMixin):
    pass


class DirectorMixin:
    _director: Director

    def init_director(self, settings: dict, api: Api):
        self._director = Director(settings=settings, api=api)

    @property
    def director(self):
        return self._director

    def complete(self):
        # TODO change to make terminate happen after timeout if specified or something.
        self.director.complete()

    def terminate(self):
        self._director.terminate()


class DatabaseMixin:
    _db: SqliteDatabase = SqliteDatabase(
        Path.home() / ".seclea" / "seclea_ai.db",
        thread_safe=True,
        pragmas={"journal_mode": "wal"},
    )

    @property
    def db(self):
        return self.db

class DatasetMixin:
    ...
class ModelMixin:
    ...


class TrainingRunMixin:
    _model:str
    _project:str
    ...
class DatasetManager(APIMixin):
    ...

    # def upload_dataset(
    #         self,
    #         dataset: Tracked,
    #         dataset_name: str,
    #         metadata: Dict,
    #         transformations: List[DatasetTransformation] = None,
    # ) -> None:
    #     """
    #     Uploads a dataset.
    #
    #     :param dataset: DataFrame, Path or list of paths to the dataset.
    #         If a list then they must be split by row only and all
    #         files must contain column names as a header line.
    #
    #     :param dataset_name: The name of the dataset.
    #
    #     :param metadata: Any metadata about the dataset. Note that if using a Path or list of Paths then if there is an
    #         index that you use when loading the data, it must be specified in the metadata.
    #
    #     :param transformations: A list of DatasetTransformation's.
    #
    #                     If your Dataset is large try call this function more often with less DatasetTransformations
    #                     as the function currently requires no. DatasetTransformations x Dataset size memory to function.
    #
    #                     See DatasetTransformation for more details.
    #
    #     :return: None
    #
    #     Example:: TODO update docs
    #         >>> seclea = SecleaAI(project_name="Test Project")
    #         >>> dataset = pd.read_csv("/test_folder/dataset_file.csv")
    #         >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
    #         >>> seclea.upload_dataset(dataset=dataset, dataset_name="Multifile Dataset", metadata=dataset_metadata)
    #
    #     Example with file::
    #
    #         >>> seclea.upload_dataset(dataset="/test_folder/dataset_file.csv", dataset_name="Test Dataset", metadata={})
    #         >>> seclea = SecleaAI(project_name="Test Project", organization="Test Organization")
    #
    #     Assuming the files are all in the /test_folder/dataset directory.
    #     Example with multiple files::
    #
    #         >>> files = os.listdir("/test_folder/dataset")
    #         >>> seclea = SecleaAI(project_name="Test Project")
    #         >>> dataset_metadata = {"index": "TransactionID", "outcome_name": "isFraud", "continuous_features": ["TransactionDT", "TransactionAmt"]}
    #         >>> seclea.upload_dataset(dataset=files, dataset_name="Multifile Dataset", metadata=dataset_metadata)
    #
    #
    #     """
    #     # processing the final dataset - make sure it's a DataFrame
    #
    #     # TODO replace with dataset_hash fn
    #     dataset_id = dataset, self._project_id
    #
    #     if transformations is not None:
    #         parent = Tracked(self._assemble_dataset(*transformations[0].raw_data_kwargs.values()))
    #
    #         #####
    #         # Validate parent exists and get metadata - check how often on portal, maybe remove?
    #         #####
    #         parent_dset_id = parent.object_manager.hash(parent, self._project_id)
    #         # check parent exists - check local db if not else error.
    #         try:
    #             res = self._api.get_dataset(
    #                 dataset_id=str(parent_dset_id),
    #                 organization_id=self._organization,
    #                 project_id=self._project_id,
    #             )
    #         except NotFoundError:
    #             # check local db
    #             self._db.connect()
    #             parent_record = Record.get_or_none(Record.key == parent_dset_id)
    #             self._db.close()
    #             if parent_record is not None:
    #                 parent_metadata = parent_record.dataset_metadata
    #             else:
    #                 raise AssertionError(
    #                     "Parent Dataset does not exist on the Platform or locally. Please check your arguments and "
    #                     "that you have uploaded the parent dataset already"
    #                 )
    #         else:
    #             parent_metadata = res.json()["metadata"]
    #         #####
    #
    #         upload_queue = self._generate_intermediate_datasets(
    #             transformations=transformations,
    #             dataset_name=dataset_name,
    #             dataset_id=dataset.object_manager.hash(dataset, self._project_id),
    #             user_metadata=metadata,
    #             parent=parent,
    #             parent_metadata=parent_metadata,
    #         )
    #
    #         # upload all the datasets and transformations.
    #         for up_kwargs in upload_queue:
    #             up_kwargs["project"] = self._project_id
    #             # add to storage and sending queues
    #             if up_kwargs["entity"] == "dataset":
    #                 self._director.store_entity(up_kwargs)
    #             self._director.send_entity(up_kwargs)
    #         return
    #
    #     # this only happens if this has no transformations ie. it is a Raw Dataset.
    #
    #     # validation
    #     features = list(getattr(dataset, 'columns', []))
    #     categorical_features = list(set(features) - set(metadata.get("continuous_features", [])))
    #     categorical_values = [{col: dataset[col].unique().tolist()} for col in categorical_features]
    #     metadata_defaults_spec = dict(
    #         continuous_features=[],
    #         outcome_name=None,
    #         num_samples=len(dataset),
    #         favourable_outcome=None,
    #         unfavourable_outcome=None,
    #         dataset_type=self._get_dataset_type(dataset),
    #         index=0 if dataset.index.name is None else dataset.index.name,
    #         split=None,
    #         features=features,
    #         categorical_features=categorical_features,
    #         categorical_values=categorical_values
    #     )
    #     metadata = {**metadata_defaults_spec, **metadata}
    #
    #     # create local db record.
    #     # TODO make lack of parent more obvious??
    #     self._db.connect()
    #     dataset_record = Record.create(
    #         entity="dataset",
    #         status=RecordStatus.IN_MEMORY.value,
    #         key=dataset.object_manager.hash(dataset, self._project_id),
    #         dataset_metadata=metadata,
    #     )
    #     self._db.close()
    #
    #     # New arch
    #     dataset_upload_kwargs = {
    #         "entity": "dataset",
    #         "record_id": dataset_record.id,
    #         "dataset": dataset,
    #         "dataset_name": dataset_name,
    #         "dataset_id": dataset_id,
    #         "metadata": metadata,
    #         "project": self._project_id,
    #     }
    #     # add to storage and sending queues
    #     self._director.store_entity(dataset_upload_kwargs)
    #     self._director.send_entity(dataset_upload_kwargs)
    #

class ModelManager(APIMixin):
    ...
    # def _set_model(self, model_name: str, framework: str) -> int:
    #     """
    #     Set the model for this session.
    #     Checks if it has already been uploaded. If not it will upload it.
    #
    #     :param model_name: The name for the architecture/algorithm. eg. "GradientBoostedMachine" or "3-layer CNN".
    #
    #     :return: int The model id.
    #
    #     :raises: ValueError - if the framework is not one of the supported frameworks or if there is an issue uploading
    #      the model.
    #     """
    #     res = self._api.get_models(
    #         organization_id=self._organization,
    #         project_id=self._project_id,
    #         name=model_name,
    #         framework=framework,
    #     )
    #     models = res.json()
    #     # not checking for more because there is a unique constraint across name and framework on backend.
    #     if len(models) == 1:
    #         return models[0]["id"]
    #     # if we got here that means that the model has not been uploaded yet. So we upload it.
    #     res = self._api.upload_model(
    #         organization_id=self._organization,
    #         project_id=self._project_id,
    #         model_name=model_name,
    #         framework_name=framework,
    #     )
    #     # TODO find out if this checking is ever needed - ie does it ever not return the created model object?
    #     try:
    #         model_id = res.json()["id"]
    #     except KeyError:
    #         traceback.print_exc()
    #         resp = self._api.get_models(
    #             organization_id=self._organization,
    #             project_id=self._project_id,
    #             name=model_name,
    #             framework=framework,
    #         )
    #         model_id = resp.json()[0]["id"]
    #     return model_id
    #

class TraininRunManager(APIMixin):
    _training_run: TrainingRunMixin


    # def upload_training_run(
    #         self,
    #         model: Tracked,
    #         train_dataset: DataFrame,
    #         test_dataset: DataFrame = None,
    #         val_dataset: DataFrame = None,
    # ) -> None:
    #     # validate the splits? maybe later when we have proper Dataset class to manage these things.
    #     dataset_ids = [
    #         dataset_hash(dataset, self._project_id)
    #         for dataset in [train_dataset, test_dataset, val_dataset]
    #         if dataset is not None
    #     ]
    #
    #     # Model stuff
    #     model_name = model.__class__.__name__
    #     # check the model exists upload if not TODO convert to add to queue
    #     model_type_id = self._set_model(
    #         model_name=model_name, framework=model.object_manager.framework
    #     )
    #
    #     # check the latest training run TODO extract all this stuff
    #     training_runs_res = self._api.get_training_runs(
    #         project_id=self._project_id,
    #         organization_id=self._organization,
    #         model=model_type_id,
    #     )
    #     training_runs = training_runs_res.json()
    #
    #     # Create the training run name
    #     largest = -1
    #     for training_run in training_runs:
    #         num = int(training_run["name"].split(" ")[2])
    #         if num > largest:
    #             largest = num
    #     training_run_name = f"Training Run {largest + 1}"
    #
    #     # extract params from the model
    #     params = model.object_manager.get_params(model)
    #
    #     # search for datasets in local db? Maybe not needed..
    #
    #     # create record
    #     self._db.connect()
    #     training_run_record = Record.create(
    #         entity="training_run", status=RecordStatus.IN_MEMORY.value
    #     )
    #
    #     # sent training run for upload.
    #     training_run_details = {
    #         "entity": "training_run",
    #         "record_id": training_run_record.id,
    #         "project": self._project_id,
    #         "training_run_name": training_run_name,
    #         "model_id": model_type_id,
    #         "dataset_ids": dataset_ids,
    #         "params": params,
    #     }
    #     self._director.send_entity(training_run_details)
    #
    #     # create local db record
    #     model_state_record = Record.create(
    #         entity="model_state",
    #         status=RecordStatus.IN_MEMORY.value,
    #         dependencies=[training_run_record.id],
    #     )
    #     self._db.close()
    #
    #     # send model state for save and upload
    #     # TODO make a function interface rather than the queue interface. Need a response to confirm it is okay.
    #     model_state_details = {
    #         "entity": "model_state",
    #         "record_id": model_state_record.id,
    #         "model": model,
    #         "sequence_num": 0,
    #         "final": True,
    #         "model_manager": model.object_manager.framework,  # TODO move framework to sender.
    #     }
    #     self._director.store_entity(model_state_details)
    #     self._director.send_entity(model_state_details)

class ProjectManager(APIMixin):
    _project: ProjectMixin

    @property
    def project(self):
        return self._project

    def init_project(self, project_name: str, organization_name: str):
        self.project.organization = OrganizationMixin(organization_name)
        self._project.name = project_name
        try:
            resp = self.api.get_project(self._project)
        except NotFoundError:
            resp = self._api.upload_project(self._project)
            if not resp.status_code == '201':
                raise Exception(resp.reason)
        self.project.deserialize(resp.json())


class UserManager(APIMixin):
    _user: UserMixin

    @property
    def user(self):
        return self._user

    def login(self) -> None:
        """
        Override login, this also overwrites the stored credentials in ~/.seclea/config.
        Note. In some circumstances the password will be echoed to stdin. This is not a problem in Jupyter Notebooks
        but may appear in scripting usage.

        :return: None

        Example::

        """
        self.api.authenticate(username=self.user.username, password=self.user.password)


class SecleaSessionMixin(UserManager, ProjectManager, SerializerMixin, MetadataMixin, ToFileMixin):
    _platform_url: str
    _auth_url: str
    _cache_path: str = ".seclea/cache"
    _offline: bool = False

    @property
    def offline(self):
        return self._offline

    @property
    def platform_url(self):
        return self._platform_url

    @property
    def auth_url(self):
        return self._auth_url

    def init_session(self, username: str,
                     password: str,
                     project_name: str,
                     organization_name: str,
                     project_root: str = ".",
                     platform_url: str = "https://platform.seclea.com",
                     auth_url: str = "https://auth.seclea.com"):
        """
        @param username: seclea platform username
        @param password: seclea platform password
        @param project_root: The path to the root of the project. Default: "."
        @param project_name:
        @param organization_name:
        @param platform_url:
        @param auth_url:
        @return:
        """
        self.file_name = self.project.name
        self.path = os.path.join(project_root, self._cache_path)
        self.user.username = username,
        self.user.password = password
        self._auth_url = auth_url
        self._platform_url = platform_url
        self.api = Api(self.serialize())
        self.init_project(project_name, organization_name)
        self.cache_session()

    def cache_session(self):
        self.metadata.update(self.serialize())
        self.save_metadata(self.full_path)

    def serialize(self):
        return {"platform_url": self.platform_url,
                "auth_url": self.auth_url,
                "offline": self.offline,
                "project": self.project.serialize(),
                "user": self.user.serialize()
                }

    def deserialize(self, obj: dict):
        self._platform_url = obj.get('platform_url')
        self._auth_url = obj.get('auth_url')
        self.project.deserialize(obj.get('project'))
        self.user.deserialize(obj.get('user'))

    def __hash__(self):
        pass
