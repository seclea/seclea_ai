import json
import os.path

from requests import Response

from seclea_ai.lib.seclea_utils.core.transmission import Transmission

# from ..errors import throws_api_err
root = "/collection/datasets"


def test_json_valid(d):
    d = json.dumps(d)
    json.loads(d)
    pass


# @throws_api_err
def post_dataset(
    transmission: Transmission,
    dataset_file_path: str,
    project_pk: str,
    organization_pk: str,
    name: str,
    metadata: dict,
    dataset_pk: str,
    parent_dataset_hash: str = None,
) -> Response:
    """
    @param transmission:
    @param dataset_file_path:
    @param project_pk:
    @param organization_pk:
    @param name:
    @param metadata:
    @param dataset_pk:
    @param parent_dataset_hash:
    @return:
    """
    dataset_queryparams = {"project": project_pk, "organization": organization_pk}
    test_json_valid(metadata)

    with open(dataset_file_path, "rb") as f:
        dataset_obj = {
            "project": (None, project_pk),
            "name": (None, name),
            "metadata": (None, json.dumps(metadata), "application/json"),
            "hash": (None, str(dataset_pk)),
            "parent_hash": (None, parent_dataset_hash),
            "dataset": (os.path.basename(dataset_file_path), f),
        }
        res = transmission.send_file(
            url_path=f"{root}",
            obj=dataset_obj,
            query_params=dataset_queryparams,
        )
    return res


def get_dataset(
    transmission: Transmission, project_pk: str, organization_pk: str, **filter_kwargs
) -> Response:
    """
    @param transmission:
    @param dataset_file_path:
    @param project_pk:
    @param organization_pk:
    @param name:
    @param metadata:
    @param dataset_pk:
    @param parent_dataset_hash:
    @return:
    """
    dataset_queryparams = {"project": project_pk, "organization": organization_pk, **filter_kwargs}

    res = transmission.get(
        url_path=f"{root}",
        query_params=dataset_queryparams,
    )
    return res
