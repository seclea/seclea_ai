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
