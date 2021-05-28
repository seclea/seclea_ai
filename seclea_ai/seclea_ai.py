import base64
import inspect
import marshal
import os
import types

from seclea_utils.auth.token_manager import update_token
from seclea_utils.data.transmission.requests_wrapper import RequestWrapper
from seclea_utils.models.sklearn.SkLearnModelManager import SkLearnModelManager


class SecleaAI:
    def __init__(
        self,
        plat_url="https://platform.seclea.com",
        auth_url="https://auth.seclea.com",
        project_name=None,
    ):
        self.s = SkLearnModelManager()
        self.trans_auth = RequestWrapper()
        self.s.manager.trans.server_root = plat_url
        self.trans_auth.server_root = auth_url
        self.username = None
        self.password = None
        self.project_name = project_name

    def login(
        self,
        username,
        password,
    ):
        print("LOGGIN IN: ")
        self.username = username
        self.password = password
        credentials = {"username": self.username, "password": self.password}
        update_token(
            trans_plat=self.s.manager.trans, trans_auth=self.trans_auth, credentials=credentials
        )
        print(self.s.manager.trans.headers)
        print(self.trans_auth.headers)

    def create_project(self, description, project_name: str = None):
        """

        :param project_name:
        :param description:
        :return:
        """
        # check for overwriting project name
        if project_name is not None and self.project_name is not None:
            print("Project name is specified twice, this is probably a bug")
        # check for no project name at all
        if project_name is None and self.project_name is None:
            print("No project name specified, please provide one")
        # check for just using initialised project name.
        if project_name is None:
            project_name = self.project_name

        self.s.manager.trans.url_path = "/collection/projects"
        res = self.s.manager.trans.send_json(
            {"name": project_name, "created_by": self.username, "description": description}
        )
        # check for already created
        self.handle_response(res, "Error creating project: ")
        self.project_name = project_name

    def handle_response(self, res, msg):
        try:
            if not res.ok:
                print(f"{msg}")
            print(res.status_code)
            print(res.data)

        except Exception as e:
            print("error upload: ", e)

    def encode_func(self, func):
        return base64.b64encode(marshal.dumps(func.__code__)).decode("ascii")

    def decode_func(self, func):
        try:
            code = marshal.loads(base64.b64decode(func))  # nosec
            f = types.FunctionType(code, globals(), "transformation1")
        except Exception as e:
            f = e
        return f

    def upload_transformations(self, transformations: list, training_run_pk: str):
        self.s.manager.trans.query_params = {}
        self.s.manager.trans.url_path = "/collection/dataset-transformations"
        for idx, trans in enumerate(transformations):
            data = {
                "name": trans.__name__,
                "code_raw": inspect.getsource(trans),
                "code_encoded": self.encode_func(trans),
                "order": idx,
                "training_run": training_run_pk,
            }
            res = self.s.manager.trans.send_json(data)
            self.handle_response(res, f"upload transformation err: {data}")

    def load_transformations(self, training_run_pk: str):
        """
        Exects a list of code_encoded as set by upload_transformations.
        """
        self.s.manager.trans.url_path = "/collection/dataset-transformations"
        self.s.manager.trans.query_params = {"training_run": training_run_pk}
        res = self.s.manager.trans.get()
        print(list(map(lambda x: x["code_encoded"], res.json())))
        transformations = list(map(lambda x: x["code_encoded"], res.json()))
        return list(map(self.decode_func, transformations))

    def upload_dataset(self, dataset_path: str, dataset_id: str, metadata: dict):
        """

        :param dataset_path:
        :param dataset_id:
        :param metadata:
        :return:
        """
        if self.project_name is None:
            raise Exception("You need to create a project before uploading a dataset")
        self.s.manager.trans.url_path = "/collection/datasets"
        dataset_queryparams = {
            "project": self.project_name,
            "identifier": dataset_id,
            "metadata": metadata,
        }
        res = self.s.manager.send_file(path=dataset_path, server_query_params=dataset_queryparams)
        self.handle_response(res, "Error uploading dataset: ")

    def upload_training_run(
        self, model, dataset_id: str, training_run_id: str, metadata: dict, sequence_no=0
    ):
        """

        :param model:
        :param dataset_id: "test-dataset-0"
        :param training_run_id: "training-run-0"
        :param metadata:  {"type": "GradientBoostedClassifier"}
        :param sequence_no: 0
        :return:
        """
        if self.project_name is None:
            raise Exception("You need to create a project before uploading a training run")
        self.s.manager.trans.url_path = "/collection/training-runs"
        self.s.manager.trans.send_json(
            {
                "project": self.project_name,
                "dataset": dataset_id,
                "identifier": training_run_id,
                "metadata": metadata,
            }
        )
        print("Uploading model")

        try:
            os.makedirs(f".seclea/{self.project_name}/{training_run_id}")
        except FileExistsError:
            print("Folder already exists, continuing")
            pass
        save_path = self.s.save_model(model, f".seclea/{self.project_name}/{training_run_id}/model")

        self.s.manager.trans.url_path = f"/collection/training-runs/{training_run_id}/states"
        self.s.manager.send_file(
            save_path,
            {
                "sequence_num": str(sequence_no),
                "training_run": training_run_id,
            },
        )
