import inspect
import json
import os
from getpass import getpass
from pathlib import Path

from seclea_utils.data.compression import Zstd
from seclea_utils.data.manager import Manager
from seclea_utils.data.transformations import decode_func, encode_func
from seclea_utils.data.transmission import RequestWrapper
from seclea_utils.models.model_management import SkLearnModelManager

from seclea_ai.exceptions import AuthenticationError


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
        self._trans_auth = RequestWrapper(auth_url)
        self._username = None
        self._access = None
        self.project_name = project_name
        self.project_exists = False
        if not os.path.isfile(os.path.join(Path.home(), ".seclea/config")):
            try:
                os.mkdir(
                    os.path.join(Path.home(), ".seclea"), mode=0o660
                )  # set mode to allow user and group rw only
            except FileExistsError:
                # do nothing.
                pass
            self.login()
        else:
            self._refresh_token()
        # here check the project exists and call create if not.
        res = self.s.manager.trans.get(f"/collection/projects/{self.project_name}")
        if res.status_code == 200:
            self.project_exists = True
        else:
            self._create_project()

    def login(self):
        self._username = input("Username: ")
        password = getpass("Password: ")
        credentials = {"username": self._username, "password": password}
        response = self._trans_auth.send_json(url_path="/api/token/obtain/", obj=credentials)
        try:
            response_content = json.loads(response.content.decode("utf-8"))
        except Exception as e:
            print(e)
            raise json.decoder.JSONDecodeError("INVALID CREDENTIALS: ", str(credentials), 1)
        self._access = response_content.get("access")
        if self._access is not None:
            self.s.manager.trans.headers = {"Authorization": f"Bearer {self._access}"}
            # note from this api access and refresh are returned together. Something to be aware of though.
            # TODO refactor when adding more to config
            with open(os.path.join(Path.home(), ".seclea/config"), "w+") as f:
                f.write(
                    json.dumps(
                        {"refresh": response_content.get("refresh"), "username": self._username}
                    )
                )
        else:
            raise AuthenticationError(
                f"There was some issue logging in: {response.status_code} {response.text}"
            )

    def _refresh_token(self):
        with open(os.path.join(Path.home(), ".seclea/config"), "r") as f:
            config = json.loads(f.read())
        try:
            refresh = config["refresh"]
            self._username = config["username"]
        except KeyError as e:
            print(e)
            # refresh token missing, prompt and login
            return self.login()
        response = self._trans_auth.send_json(
            url_path="/api/token/refresh/", obj={"refresh": refresh}
        )
        if not response.ok:
            self.handle_response(res=response, msg="There was an issue with the refresh token")
            return self.login()
        else:
            try:
                response_content = json.loads(response.content.decode("utf-8"))
                self._access = response_content.get("access")
                if self._access is not None:
                    self.s.manager.trans.headers = {"Authorization": f"Bearer {self._access}"}
                else:
                    return self.login()
            except Exception as e:
                print(e)
                self.login()

    def _create_project(self):
        """

        :return:
        """
        res = self.s.manager.trans.send_json(
            url_path="/collection/projects",
            obj={
                "name": self.project_name,
                "created_by": self._username,
                "description": "Please add a description..",
            },
        )
        if res.status_code == 201:
            self.project_exists = True
        else:
            self.handle_response(
                res,
                "There was an issue creating the project, this may be expected if the project already exists",
            )

    @staticmethod
    def handle_response(res, msg):
        if not res.ok:
            print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")

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
            self.handle_response(res, f"upload transformation err: {data}")

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
        if not self.project_exists:
            raise Exception("You need to create a project before uploading a dataset")
        dataset_queryparams = {
            "project": self.project_name,
            "identifier": dataset_id,
            "metadata": metadata,
        }
        res = self.s.manager.trans.send_file(
            url_path="/collection/datasets",
            file_path=dataset_path,
            query_params=dataset_queryparams,
        )
        self.handle_response(res, "Error uploading dataset: ")

    def upload_training_run(self, training_run_id: str, dataset_id: str, metadata: dict):
        """

        :param dataset_id: "test-dataset-0"
        :param training_run_id: "training-run-0"
        :param metadata:  {"type": "GradientBoostedClassifier"}
        :return:
        """
        if not self.project_exists:
            raise Exception("You need to create a project before uploading a training run")
        res = self.s.manager.trans.send_json(
            url_path="/collection/training-runs",
            obj={
                "project": self.project_name,
                "dataset": dataset_id,
                "identifier": training_run_id,
                "metadata": metadata,
            },
        )
        self.handle_response(res, "There was an issue uploading the training run")

    def upload_model_state(self, model, training_run_id, sequence_num, final=False):
        try:
            os.makedirs(os.path.join(Path.home(), f".seclea/{self.project_name}/{training_run_id}"))
        except FileExistsError:
            print("Folder already exists, continuing")
            pass
        save_path = self.s.save_model(
            model,
            os.path.join(
                Path.home(), f".seclea/{self.project_name}/{training_run_id}/model-{sequence_num}"
            ),
        )

        res = self.s.manager.trans.send_file(
            url_path="/collection/model-states",
            file_path=save_path,
            query_params={
                "sequence_num": sequence_num,
                "training_run": training_run_id,
                "project": self.project_name,
                "final_state": final,
            },
        )
        self.handle_response(res, "There was and issue uploading the model state")
        if res.status_code == 201:
            os.remove(save_path)
