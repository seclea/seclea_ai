import inspect
import os
from pathlib import Path

from requests import Response
from seclea_utils.data.compression import Zstd
from seclea_utils.data.manager import Manager
from seclea_utils.data.transformations import decode_func, encode_func
from seclea_utils.data.transmission import RequestWrapper
from seclea_utils.models.model_management import SkLearnModelManager

from seclea_ai.authentication import AuthenticationService


def handle_response(res: Response, msg):
    if not res.ok:
        print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")


class SecleaAI:
    def __init__(
        self,
        project_name,
        plat_url="https://platform.seclea.com",
        auth_url="https://auth.seclea.com",
    ):
        self.s = SkLearnModelManager(
            Manager(compression=Zstd(), transmission=RequestWrapper(server_root_url=plat_url))
        )
        self._auth_service = AuthenticationService(RequestWrapper(auth_url))
        self._username, auth_creds = self._auth_service.handle_auth()
        self.s.manager.trans.headers = auth_creds
        self._project_name = project_name
        self._project_exists = False

        # here check the project exists and call create if not.
        res = self.s.manager.trans.get(f"/collection/projects/{self._project_name}")
        if res.status_code == 200:
            self._project_exists = True
        else:
            self._create_project()

    def login(self) -> None:
        """
        Override login, this also overwrites the stored credentials in ~/.seclea/config.
        :return: None
        """
        self._username, auth_creds = self._auth_service.login()
        self.s.manager.trans.headers = auth_creds

    def _create_project(self) -> None:
        """
        Creates a new project
        :return:
        """
        res = self.s.manager.trans.send_json(
            url_path="/collection/projects",
            obj={
                "name": self._project_name,
                "created_by": self._username,
                "description": "Please add a description..",
            },
        )
        if res.status_code == 201:
            self._project_exists = True
        else:
            handle_response(
                res,
                "There was an issue creating the project.",
            )

    def upload_transformations(self, transformations: list, training_run_pk: str):
        for idx, trans in enumerate(transformations):
            data = {
                "name": trans.__name__,
                "code_raw": inspect.getsource(trans),
                "code_encoded": encode_func(trans),
                "order": idx,
                "training_run": training_run_pk,
            }
            res = self.s.manager.trans.send_json(
                url_path="/collection/dataset-transformations", obj=data
            )
            handle_response(res, f"upload transformation err: {data}")

    def load_transformations(self, training_run_pk: str):
        """
        Expects a list of code_encoded as set by upload_transformations.
        """
        res = self.s.manager.trans.get(
            url_path="/collection/dataset-transformations",
            query_params={"training_run": training_run_pk},
        )
        transformations = list(map(lambda x: x["code_encoded"], res.json()))
        return list(map(decode_func, transformations))

    def upload_dataset(self, dataset_path: str, dataset_id: str, metadata: dict):
        """
        TODO add pii check to here before upload.
        :param dataset_path:
        :param dataset_id:
        :param metadata:
        :return:
        """
        if not self._project_exists:
            raise Exception("You need to create a project before uploading a dataset")
        dataset_queryparams = {
            "project": self._project_name,
            "identifier": dataset_id,
            "metadata": metadata,
        }
        res = self.s.manager.trans.send_file(
            url_path="/collection/datasets",
            file_path=dataset_path,
            query_params=dataset_queryparams,
        )
        handle_response(res, "Error uploading dataset: ")

    def upload_training_run(self, training_run_id: str, dataset_id: str, metadata: dict):
        """

        :param dataset_id: "test-dataset-0"
        :param training_run_id: "training-run-0"
        :param metadata:  {"type": "GradientBoostedClassifier"}
        :return:
        """
        if not self._project_exists:
            raise Exception("You need to create a project before uploading a training run")
        res = self.s.manager.trans.send_json(
            url_path="/collection/training-runs",
            obj={
                "project": self._project_name,
                "dataset": dataset_id,
                "identifier": training_run_id,
                "metadata": metadata,
            },
        )
        handle_response(res, "There was an issue uploading the training run")

    def upload_model_state(self, model, training_run_id, sequence_num, final=False):
        try:
            os.makedirs(
                os.path.join(Path.home(), f".seclea/{self._project_name}/{training_run_id}")
            )
        except FileExistsError:
            print("Folder already exists, continuing")
            pass
        save_path = self.s.save_model(
            model,
            os.path.join(
                Path.home(), f".seclea/{self._project_name}/{training_run_id}/model-{sequence_num}"
            ),
        )

        res = self.s.manager.trans.send_file(
            url_path="/collection/model-states",
            file_path=save_path,
            query_params={
                "sequence_num": sequence_num,
                "training_run": training_run_id,
                "project": self._project_name,
                "final_state": final,
            },
        )
        handle_response(res, "There was and issue uploading the model state")
        if res.status_code == 201:
            os.remove(save_path)
