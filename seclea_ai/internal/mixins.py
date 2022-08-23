from __future__ import annotations

import os
from pathlib import Path
from typing import List

from peewee import SqliteDatabase

from seclea_ai.internal.api.api_interface import PlatformApi
from seclea_ai.internal.director import Director
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management import Tracked
from seclea_ai.lib.seclea_utils.object_management.mixin import ToFileMixin, SerializerMixin, MetadataMixin, User, \
    Project, Organization
from seclea_ai.transformations import DatasetTransformation

from seclea_ai.lib.seclea_utils.object_management.mixin import Dataset


class APIMixin:
    _api: PlatformApi

    @property
    def api(self) -> PlatformApi:
        return self._api

    @api.setter
    def api(self, val: PlatformApi):
        self._api = val


class DirectorMixin:
    _director: Director
    db: SqliteDatabase

    def init_director(self, cache_dir):
        self._director = Director(cache_dir=cache_dir)

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
        return self._db


class TrainingRunMixin:
    _model: str
    _project: str
    ...


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


class OrganizationManager:
    api: PlatformApi
    _organization: Organization

    @property
    def organization(self):
        if getattr(self, '_organization', None) is None:
            self._organization = Organization()
        return self._organization

    def init_organization(self, uuid=None, name=None):
        self._organization = self.api.organizations.get_one_from_list(uuid=uuid, name=name, params=dict())
        print(self._organization)


class ProjectManager:
    api: PlatformApi
    _project: Project

    @property
    def project(self) -> Project:
        if getattr(self, '_project', None) is None:
            self._project = Project()
        return self._project

    def init_project(self, project_name: str, organization_id: Organization.uuid):
        try:
            self._project = self.api.projects.get_one_from_list(name=project_name, organization=organization_id)
            return
        except Exception as e:
            print(f'Error getting project {e} attemptin initialization of new project:')
        try:
            self._project = self.api.projects.create({'name': project_name, 'organization': organization_id},
                                                     {'organization': organization_id})
        except Exception as e:
            print(f"Failed to init new project: {e}")
            raise


class UserManager:
    api: APIMixin.api
    _user: User

    @property
    def user(self):
        if getattr(self, '_user', None) is None:
            self._user = User()
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


class DatasetManager:
    api: PlatformApi
    organization: Organization
    db: DatabaseMixin.db
    director: DirectorMixin.director
    project: Project

    def upload_dataset(self, dataset: Tracked, transformations: List[DatasetTransformation] = None):
        """
        # assemble all necessary metadata,
        # compress into file
        # upload

        """
        # validation
        features = list(getattr(dataset, 'columns', []))
        categorical_features = list(set(features) - set(dataset.object_manager.metadata.get("continuous_features", [])))
        categorical_values = [{col: dataset[col].unique().tolist()} for col in categorical_features]
        metadata_defaults_spec = dict(
            continuous_features=[],
            outcome_name=None,
            num_samples=len(dataset),
            favourable_outcome=None,
            unfavourable_outcome=None,
            index=0 if dataset.index.name is None else dataset.index.name,
            split=None,
            features=features,
            categorical_features=categorical_features,
            categorical_values=categorical_values
        )
        metadata = {**metadata_defaults_spec, **dataset.object_manager.metadata}
        params = {
            "organization": self.organization.uuid,
            "project": self.project.uuid
        }
        dset = Dataset(
            metadata=metadata,
            name=dataset.object_manager.file_name,
            project=self.project.uuid,
            description="some description",
        )
        # TODO: The record should only be created if it doesn't already exist...
        self.db.connect()
        dataset_record = Record.create(
            object_ser=dset.serialize(),
            status=RecordStatus.IN_MEMORY.value,
        )
        # TODO remove this,
        dset.uuid = dataset_record.id
        self.db.close()
        self.director.cache_upload_object(obj_tracked=dataset, obj_bs=dset, api=self.api.datasets, params=params)


class SecleaSessionMixin(UserManager, DatasetManager, OrganizationManager, ProjectManager, MetadataMixin, ToFileMixin):
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
                     auth_url: str = "https://auth.seclea.com",
                     cache_dir: str = '.'):
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
        self._cache_path = cache_dir
        self.file_name = project_name
        self.path = os.path.join(project_root, self._cache_path)
        self.user.username = username,
        self.user.password = password
        self._auth_url = auth_url
        self._platform_url = platform_url
        self.api = PlatformApi(auth_url=auth_url, platform_url=platform_url, username=username, password=password)
        self.init_organization(name=organization_name)
        self.init_project(project_name, self.organization.uuid)
        self.cache_session()

    def cache_session(self):
        # ensure path exists
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
