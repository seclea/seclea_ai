"""
Description for seclea_ai.py
"""
from __future__ import annotations

import copy
import inspect
import logging
import os
import traceback
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.errors import ParserError

from seclea_ai.internal.exceptions import AuthenticationError, NotFoundError
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.internal.mixins import DatabaseMixin, DirectorMixin, SecleaSessionMixin
from seclea_ai.lib.seclea_utils.core import encode_func
from seclea_ai.lib.seclea_utils.object_management import Tracked
from seclea_ai.transformations import DatasetTransformation

logger = logging.getLogger(__name__)
print(os.getcwd())


class SecleaAI(SecleaSessionMixin, DatabaseMixin, DirectorMixin):
    _training_run = None

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

            >>> seclea = SecleaAI(project="Test Project", project_root=".")
        """
        self.init_session(username=username,
                          password=password,
                          project_root=project_root,
                          project_name=project_name,
                          organization_name=organization,
                          platform_url=platform_url,
                          auth_url=auth_url)
        self.init_director(self.metadata, self.api)

        # TODO add username and password?
        logger.debug("Successfully Initialised SecleaAI class")


