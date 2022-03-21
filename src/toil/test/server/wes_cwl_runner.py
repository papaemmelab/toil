import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional, Dict, Any, List

from toil.test.server.wes_client import WESClient

"""
A CWL runner that submits a workflow to a WES server, waits for it to finish,
and outputs the results.

Example usage with cwltest:

```
cwltest --verbose \
    --tool=python \
    --test=cwl-v1.2-main/conformance_tests.yaml \
    -n=0-197 \
    --timeout=2400 \
    -j1 \
    -- \
    src/toil/test/server/wes_cwl_runner.py \
    --wes_endpoint=http://localhost:8080 \
    --wes_user=test \
    --wes_password=password \
    --disableCaching \
    --clean=always \
    --logDebug
```
"""

logger = logging.getLogger(__name__)


def get_deps_from_cwltool(cwl_file: str, input_file: Optional[str] = None):
    """
    Return a list of dependencies of the given workflow from cwltool.

    :param cwl_file: The CWL file.
    :param input_file: Omit to get the dependencies from the CWL file. If set,
                       this returns the dependencies from the input file.
    """

    option = '--print-input-deps' if input_file else '--print-deps'

    args = ['cwltool', option, '--relative-deps', 'cwd', cwl_file]
    if input_file:
        args.append(input_file)

    p = subprocess.run(args=args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    result = p.stdout.decode()
    if not result:
        return []

    json_result: Dict[str, Any] = json.loads(result)

    deps = []

    def get_deps(obj: Any):
        for file in obj:
            if isinstance(file, dict) and "location" in file:
                loc = file.get("location")

                # Check directory
                if file.get("class") == "Directory":
                    if file.get("listing"):
                        get_deps(file.get("listing"))
                    else:
                        # no listing, so import all files in the directory
                        for folder, _, sub_files in os.walk(loc):
                            for sub_file in sub_files:
                                deps.append(os.path.join(folder, sub_file))
                else:
                    deps.append(file.get("location"))

                # check secondaryFiles
                if "secondaryFiles" in file:
                    get_deps(file.get("secondaryFiles"))

    get_deps(json_result.get("secondaryFiles", []))
    return deps


def submit_run(client: WESClient,
               cwl_file: str,
               input_file: Optional[str] = None,
               engine_options: Optional[List[str]] = None) -> str:
    # First, get the list of files to attach to this workflow
    attachments = get_deps_from_cwltool(cwl_file, input_file)

    if input_file:
        attachments.extend(get_deps_from_cwltool(cwl_file, input_file))

    # logging.warning(f"Files to import: {attachments}")

    run_result: Dict[str, Any] = client.run(cwl_file, input_file,
                                            attachments=attachments,
                                            engine_options=engine_options)
    return run_result.get("run_id", None)


def poll_run(client: WESClient, run_id: str) -> bool:
    """ Return True if the given workflow run is in a finished state."""
    status_result = client.get_run_status(run_id)
    state = status_result.get("state")

    return state in ("COMPLETE", "CANCELING", "CANCELED", "EXECUTOR_ERROR", "SYSTEM_ERROR")


def print_logs_and_exit(client: WESClient, run_id: str) -> None:
    # cwltest checks for stdout, stderr, and return code
    data = client.get_run_log(run_id)

    outputs = json.dumps(data.get("outputs", {}), indent=4)
    exit_code = data.get("run_log", {}).get("exit_code", 1)

    sys.stdout.write(outputs)
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser()
    # first two positional arguments are passed in by cwltest
    parser.add_argument("cwl_file", type=str)
    parser.add_argument("input_file", type=str, nargs="?", default=None)
    # a few arguments useful for the WES runner
    parser.add_argument("--wes_endpoint", default="http://localhost:8080",
                        help="The http(s) URL of the WES server.  (default: %(default)s)")
    parser.add_argument("--wes_user", default=None,
                        help="Username to use for basic authentication to the WES server.")
    parser.add_argument("--wes_password", default=None,
                        help="Password to use for basic authentication to the WES server.")
    # the rest of the arguments are passed as engine options to the WES server
    options, rest = parser.parse_known_args()

    cwl_file = options.cwl_file
    input_file = options.input_file

    # Initialize client and run the workflow
    endpoint = options.wes_endpoint
    # logger.warning(f"Using WES endpoint: {endpoint.scheme}://{endpoint.netloc}")

    wes_user = options.wes_user
    wes_password = options.wes_password
    client = WESClient(base_url=endpoint,
                       auth=(wes_user, wes_password) if wes_user and wes_password else None)

    run_id = submit_run(client, cwl_file, input_file, engine_options=rest)
    assert run_id

    # logger.warning(f"Submitted run. Waiting for {run_id} to finish...")

    done = False
    while not done:
        time.sleep(1)
        done = poll_run(client, run_id)

    print_logs_and_exit(client, run_id)


if __name__ == '__main__':
    main()
