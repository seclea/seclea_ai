from seclea_ai.lib.seclea_utils.core.transmission import Transmission
import os
import aiohttp

root = "/collection/model-states"


async def async_post_model_state(
    transmission: Transmission,
    model_state_file_path: str,
    organization_pk: str,
    project_pk: str,
    training_run_pk: str,
    sequence_num: int,
    final_state,
    delete=False,
):
    """

    @param transmission:
    @param model_state_file_path:
    @param organization_pk:
    @param project_pk:
    @param training_run_pk:
    @param sequence_num:
    @param final_state:
    @param delete:
    @return:
    """
    query_params = {
            "organization": organization_pk,
            "project": str(project_pk),
            "sequence_num": sequence_num,
            "training_run": str(training_run_pk),
            "final_state": str(final_state),
    }

    res = await upload_to_server(f"{root}", model_state_file_path, query_params, transmission, delete)

    return res

# upload model states to portal asynchronously
async def upload_to_server(url_path, model_state_file_path, query_params, transmission, delete_file):
    with open(model_state_file_path, 'rb') as f:
        headers = transmission.headers
        headers["Content-Disposition"] = f"attachment; filename={url_path}"
        request_path = f"{transmission._server_root}{url_path}"
        try:
           async with aiohttp.ClientSession(cookies=transmission.cookies, headers=headers) as session:
                async with session.post(request_path, data={'file': f}, params=query_params) as response:
                    if delete_file:
                        os.remove(model_state_file_path)
                    return response
        except Exception as e:
            print(e)
