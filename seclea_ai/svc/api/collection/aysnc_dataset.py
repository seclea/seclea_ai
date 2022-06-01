import json
import os
import aiohttp

from seclea_ai.lib.seclea_utils.core.transmission import Transmission

root = "/collection/datasets"

async def async_post_dataset(
    transmission: Transmission,
    dataset_file_path: str,
    project_pk: str,
    organization_pk: str,
    name: str,
    metadata: dict,
    dataset_hash: str,
    parent_dataset_hash: str = None,
    delete=False,
):
    """
    @param transmission:
    @param dataset_file_path:
    @param project_pk:
    @param organization_pk:
    @param name:
    @param metadata:
    @param dataset_hash:
    @param parent_dataset_hash:
    @param delete: delete dataset file
    @return:
    """
    dataset_queryparams = {
        "project": str(project_pk),
        "organization": str(organization_pk),
        "name": name,
        "metadata": json.dumps(metadata),
        "hash": str(dataset_hash),
    }

    if parent_dataset_hash and parent_dataset_hash != None:
        dataset_queryparams['parent'] = parent_dataset_hash

    res = await upload_to_server(f"{root}", dataset_file_path, dataset_queryparams, transmission, delete)

    return res

# upload dataset to portal asynchronously
async def upload_to_server(url_path, dataset_file_path, dataset_queryparams, transmission, delete_file):
    with open(dataset_file_path, 'rb') as f:
        headers = transmission.headers
        headers["Content-Disposition"] = f"attachment; filename={url_path}"
        request_path = f"{transmission._server_root}{url_path}"
        try:
           async with aiohttp.ClientSession(cookies=transmission.cookies, headers=headers) as session:
                async with session.post(request_path, data={'file': f}, params=dataset_queryparams) as response:
                    if delete_file:
                        os.remove(dataset_file_path)
                    return response
        except Exception as e:
            print(e)
