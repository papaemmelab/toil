# Modified from wes-service under Apache License, Version 2.0.
# https://github.com/common-workflow-language/workflow-service/blob/main/LICENSE
import json
import os
import requests
import logging
import ruamel.yaml

from io import BytesIO
from toil.wdl.utils import get_version as get_wdl_version
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urldefrag, urlparse


def get_version(extension: str, workflow_file: str):
    """Determines the version of a .py, .wdl, or .cwl file."""
    if extension == "py":
        return "3.8"
    elif extension == "cwl":
        with open(workflow_file) as f:
            return ruamel.yaml.safe_load(f)["cwlVersion"]
    elif extension == "wdl":
        with open(workflow_file) as f:
            return get_wdl_version(f)
    else:
        raise RuntimeError(f"Invalid workflow extension: {extension}.")


def parse_params(workflow_params: Dict[str, Any]) -> None:
    """
    Loop through files in the input workflow parameters json and make sure
    the paths do not include relative paths to parent directories.
    """
    from werkzeug.utils import secure_filename

    def secure_path(path: str) -> str:
        return os.path.join(*[str(secure_filename(p)) for p in path.split("/") if p not in ("", ".", "..")])

    def replace_paths(obj: Any) -> None:
        for file in obj:
            if isinstance(file, dict) and "location" in file:
                loc = file.get("location")
                if isinstance(loc, str) and urlparse(loc).scheme in ("file", ""):
                    file["location"] = secure_path(loc)

                if "secondaryFiles" in file:
                    replace_paths(file.get("secondaryFiles"))

    replace_paths(workflow_params.values())


def build_wes_request(workflow_url: str,
                      workflow_params_url: str,
                      attachments: List[str],
                      workflow_engine_parameters: Optional[List[str]] = None):

    # Read from the workflow_param file and parse it into a dict
    if workflow_params_url:
        with open(workflow_params_url, "r") as f:
            if workflow_params_url.endswith((".yaml", ".yml")):
                workflow_params = ruamel.yaml.safe_load(f)
            elif workflow_params_url.endswith(".json"):
                workflow_params = json.load(f)
            else:
                raise ValueError(f"Unsupported file type for workflow_params: '{os.path.basename(workflow_params_url)}'")
    else:
        workflow_params = {}
    parse_params(workflow_params)

    # Initialize the basic parameters for the run request
    wf_url, frag = urldefrag(workflow_url)
    workflow_type = wf_url.lower().split(".")[-1]  # Grab the file extension
    workflow_type_version = get_version(workflow_type, wf_url)
    data = {
        "workflow_url": os.path.basename(workflow_url),
        "workflow_params": json.dumps(workflow_params),
        "workflow_type": workflow_type,
        "workflow_type_version": workflow_type_version
    }

    # Convert engine arguments into a JSON object
    if workflow_engine_parameters:
        params = {}
        for param in workflow_engine_parameters:
            if '=' not in param:  # flags like "--logDebug"
                k, v = param, None
            else:
                k, v = param.split('=', 1)
            params[k] = v
        data["workflow_engine_parameters"] = json.dumps(params)

    # Deal with workflow attachments
    base = os.path.dirname(wf_url)
    attachments.append(wf_url)

    workflow_attachments = []
    for file in attachments:
        with open(file, "rb") as f:
            rel: str = os.path.relpath(file, base)
            if '../' in rel:
                # when inputs are in a different directory from the workflow
                rel = os.path.basename(file)
            workflow_attachments.append((rel, BytesIO(f.read())))

    return data, (("workflow_attachment", val) for val in workflow_attachments)


def wes_response(post_result):
    if post_result.status_code != 200:
        error = str(json.loads(post_result.text))
        logging.error(error)
        raise Exception(error)

    return json.loads(post_result.text)


class WESClient:
    def __init__(self, base_url: str, auth: Optional[Tuple[str, str]] = None):
        """
        :param base_url: Base URL of the WES server.
        :param auth: The username and password to use for basic authentication to the WES server.
        """
        self.base_url = base_url
        self.auth = auth

    def get_service_info(self):
        """
        Get information about Workflow Execution Service. May include
        information related (but not limited to) the workflow descriptor
        formats, versions supported, the WES API versions supported, and
        information about general the service availability.

        :return: The body of the get result as a dictionary.
        """
        post_result = requests.get(
            f"{self.base_url}/ga4gh/wes/v1/service-info",
            auth=self.auth,
        )
        return wes_response(post_result)

    def list_runs(self):
        """
        List the workflows, this endpoint will list the workflows in order of
        oldest to newest. There is no guarantee of live updates as the user
        traverses the pages, the behavior should be decided (and documented) by
        each implementation.

        :return: The body of the get result as a dictionary.
        """
        post_result = requests.get(
            f"{self.base_url}/ga4gh/wes/v1/runs", auth=self.auth
        )
        return wes_response(post_result)

    def run(self, workflow_file, jsonyaml, attachments, engine_options):
        """
        Composes and sends a post request that signals the wes server to run a workflow.

        :param str workflow_file: A local/http/https path to a cwl/wdl/python workflow file.
        :param str jsonyaml: A local path to a json or yaml file.
        :param list attachments: A list of local paths to files that will be uploaded to the server.
        :param list engine_options: A list of engine options to attach.

        :return: The body of the post result as a dictionary.
        """
        data, files = build_wes_request(workflow_file, jsonyaml, attachments, engine_options)
        post_result = requests.post(
            f"{self.base_url}/ga4gh/wes/v1/runs",
            data=data,
            files=files,
            auth=self.auth,
        )
        return wes_response(post_result)

    def cancel(self, run_id):
        """
        Cancel a running workflow.

        :param run_id: String (typically a uuid) identifying the run.
        :return: The body of the delete result as a dictionary.
        """
        post_result = requests.post(
            f"{self.base_url}/ga4gh/wes/v1/runs/{run_id}/cancel",
            auth=self.auth,
        )
        return wes_response(post_result)

    def get_run_log(self, run_id):
        """
        Get detailed info about a running workflow.

        :param run_id: String (typically a uuid) identifying the run.
        :return: The body of the get result as a dictionary.
        """
        post_result = requests.get(
            f"{self.base_url}/ga4gh/wes/v1/runs/{run_id}",
            auth=self.auth,
        )
        return wes_response(post_result)

    def get_run_status(self, run_id):
        """
        Get quick status info about a running workflow.

        :param run_id: String (typically a uuid) identifying the run.
        :return: The body of the get result as a dictionary.
        """
        post_result = requests.get(
            f"{self.base_url}/ga4gh/wes/v1/runs/{run_id}/status",
            auth=self.auth,
        )
        return wes_response(post_result)