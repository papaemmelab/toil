# Copyright (C) 2015-2021 Regents of the University of California
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os

from argparse import ArgumentParser, _ArgumentGroup
from typing import Any, Callable, List, Optional, Union, TypeVar

from toil.batchSystems.registry import (
    BATCH_SYSTEM_FACTORY_REGISTRY,
    BATCH_SYSTEMS,
    DEFAULT_BATCH_SYSTEM,
)
from toil.lib.threading import cpu_count

def set_batchsystem_options(batch_system: str, set_option: Callable) -> None:
    """
    Call set_option for all the options for the given named batch system.
    """
    batch_system_factory = BATCH_SYSTEM_FACTORY_REGISTRY[batch_system]()
    batch_system_factory.setOptions(set_option)

def add_all_batchsystem_options(parser: Union[ArgumentParser, _ArgumentGroup]) -> None:
    # Do the global cross-batch-system arguments
    parser.add_argument(
        "--batchSystem",
        dest="batchSystem",
        default=DEFAULT_BATCH_SYSTEM,
        choices=BATCH_SYSTEMS,
        help=f"The type of batch system to run the job(s) with, currently can be one "
        f"of {', '.join(BATCH_SYSTEMS)}. default={DEFAULT_BATCH_SYSTEM}",
    )
    parser.add_argument(
        "--disableHotDeployment",
        dest="disableAutoDeployment",
        action="store_true",
        default=None,
        help="Hot-deployment was renamed to auto-deployment.  Option now redirects to "
        "--disableAutoDeployment.  Left in for backwards compatibility.",
    )
    parser.add_argument(
        "--disableAutoDeployment",
        dest="disableAutoDeployment",
        action="store_true",
        default=None,
        help="Should auto-deployment of the user script be deactivated? If True, the user "
        "script/package should be present at the same location on all workers.  Default = False.",
    )
    parser.add_argument(
        "--maxLocalJobs",
        default=cpu_count(),
        help=f"For batch systems that support a local queue for housekeeping jobs "
        f"(Mesos, GridEngine, htcondor, lsf, slurm, torque).  Specifies the maximum "
        f"number of these housekeeping jobs to run on the local system.  The default "
        f"(equal to the number of cores) is a maximum of {cpu_count()} concurrent "
        f"local housekeeping jobs.",
    )
    parser.add_argument(
        "--manualMemArgs",
        default=False,
        action="store_true",
        dest="manualMemArgs",
        help="Do not add the default arguments: 'hv=MEMORY' & 'h_vmem=MEMORY' to the qsub "
        "call, and instead rely on TOIL_GRIDGENGINE_ARGS to supply alternative arguments.  "
        "Requires that TOIL_GRIDGENGINE_ARGS be set.",
    )
    parser.add_argument(
        "--runCwlInternalJobsOnWorkers",
        dest="runCwlInternalJobsOnWorkers",
        action="store_true",
        default=None,
        help="Whether to run CWL internal jobs (e.g. CWLScatter) on the worker nodes "
        "instead of the primary node. If false (default), then all such jobs are run on "
        "the primary node. Setting this to true can speed up the pipeline for very large "
        "workflows with many sub-workflows and/or scatters, provided that the worker "
        "pool is large enough.",
    )
    parser.add_argument(
        "--coalesceStatusCalls",
        dest="coalesceStatusCalls",
        action="store_true",
        default=None,
        help=(
            "Coalese status calls to prevent the batch system from being overloaded. "
            "Currently only supported for LSF. "
            "default=false"
        ),
    )

    for factory in BATCH_SYSTEM_FACTORY_REGISTRY.values():
        # All the batch systems are responsible for adding their own options
        # with the add_options class method.
        try:
            batch_system_type = factory()
        except ImportError:
            # Skip anything we can't import
            continue
        # Ask the batch system to create its options in the parser
        batch_system_type.add_options(parser)

def set_batchsystem_config_defaults(config) -> None:
    """
    Set default and environment-based options for builtin batch systems. This
    is required if a Config object is not constructed from an Options object.
    """

    # Do the global options across batch systems
    config.batchSystem = "single_machine"
    config.disableAutoDeployment = False
    config.environment = {}
    config.statePollingWait = None
    config.maxLocalJobs = cpu_count()
    config.manualMemArgs = False
    config.coalesceStatusCalls = False

    T = TypeVar('T')
    def set_option(option_name: str,
                   parsing_function: Optional[Callable[[str], T]] = None,
                   check_function: Optional[Callable[[T], None]] = None,
                   default: Optional[T] = None,
                   env: Optional[List[str]] = None) -> None:
        """
        Function to set a batch-system-defined option to its default value.
        """

        # TODO: deduplicate with Config

        option_value = default

        if env is not None:
            for env_var in env:
                # Try all the environment variables
                if option_value != default:
                    break
                option_value = os.environ.get(env_var, default)

        if option_value is not None:
            if parsing_function is not None and isinstance(option_value, str):
                option_value = parsing_function(option_value)
            if check_function is not None:
                try:
                    check_function(option_value)
                except AssertionError:
                    raise RuntimeError(f"The {option_name} option has an invalid value: {option_value}")

        setattr(config, option_name, option_value)

    for factory in BATCH_SYSTEM_FACTORY_REGISTRY.values():
        # All the batch systems are responsible for setting their own defaults
        # with their setOptions() class methods.
        try:
            batch_system_type = factory()
        except ImportError:
            # Skip anything we can't import
            continue
        # Ask the batch system to tell us all its options.
        batch_system_type.setOptions(set_option)
