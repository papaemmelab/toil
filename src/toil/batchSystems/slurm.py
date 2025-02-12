# Copyright (c) 2016 Duke Center for Genomic and Computational Biology
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
# limitations under the License.
import logging
import math
import os
import subprocess
from collections import defaultdict
from pipes import quote
from typing import Dict, List, Optional, Set, Tuple, Union

from toil.batchSystems.abstractBatchSystem import BatchJobExitReason, EXIT_STATUS_UNAVAILABLE_VALUE
from toil.batchSystems.abstractGridEngineBatchSystem import AbstractGridEngineBatchSystem
from toil.lib.conversions import bytes2human
from toil.lib.misc import CalledProcessErrorStderr, call_command
from toil.statsAndLogging import TRACE

logger = logging.getLogger(__name__)

MAX_MEMORY = 60 * 1e9
OUT_OF_MEM_RETRIES = 2

# We have a complete list of Slurm states. States not in one of these aren't
# allowed. See <https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES>

# If a job is in one of these states, Slurm can't run it anymore.
# We don't include states where the job is held or paused here;
# those mean it could run and needs to wait for someone to un-hold
# it, so Toil should wait for it.
#
# We map from each terminal state to the Toil-ontology exit reason.
TERMINAL_STATES: Dict[str, BatchJobExitReason] = {
    "BOOT_FAIL": BatchJobExitReason.LOST,
    "CANCELLED": BatchJobExitReason.KILLED,
    "COMPLETED": BatchJobExitReason.FINISHED,
    "DEADLINE": BatchJobExitReason.KILLED,
    "FAILED": BatchJobExitReason.FAILED,
    "NODE_FAIL": BatchJobExitReason.LOST,
    "OUT_OF_MEMORY": BatchJobExitReason.MEMLIMIT,
    "PREEMPTED": BatchJobExitReason.KILLED,
    "REVOKED": BatchJobExitReason.KILLED,
    "SPECIAL_EXIT": BatchJobExitReason.FAILED,
    "TIMEOUT": BatchJobExitReason.KILLED
}

# If a job is in one of these states, it might eventually move to a different
# state.
NONTERMINAL_STATES: Set[str] = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESV_DEL_HOLD",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "REQUEUED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED"
}

class SlurmBatchSystem(AbstractGridEngineBatchSystem):
    
    def __init__(self, *args, **kwargs):
        """Create a mapping table for JobIDs to JobNodes."""
        super(SlurmBatchSystem, self).__init__(*args, **kwargs)
        self.Id2Node: Dict[int, Dict] = {}
        self.resourceRetryCount = defaultdict(int)

    def issueBatchJob(self, jobDesc, job_environment=None):
        """Load the jobDesc into the JobID mapping table."""
        jobID = super(SlurmBatchSystem, self).issueBatchJob(jobDesc, job_environment)
        self.Id2Node[jobID] = jobDesc
        return jobID

    class Worker(AbstractGridEngineBatchSystem.Worker):
        
        def forgetJob(self, jobID):
            """Remove jobNode from the mapping table when forgetting."""
            self.boss.Id2Node.pop(jobID, None)
            self.boss.resourceRetryCount.pop(jobID, None)
            return super(SlurmBatchSystem.Worker, self).forgetJob(jobID)

        def getRunningJobIDs(self):
            # Should return a dictionary of Job IDs and number of seconds
            times = {}
            with self.runningJobsLock:
                currentjobs: Dict[str, int] = {str(self.batchJobIDs[x][0]): x for x in self.runningJobs}
            # currentjobs is a dictionary that maps a slurm job id (string) to our own internal job id
            # squeue arguments:
            # -h for no header
            # --format to get jobid i, state %t and time days-hours:minutes:seconds

            lines = call_command(['squeue', '-h', '--format', '%i %t %M'], quiet=True).split('\n')
            for line in lines:
                values = line.split()
                if len(values) < 3:
                    continue
                slurm_jobid, state, elapsed_time = values
                if slurm_jobid in currentjobs and state == 'R':
                    seconds_running = self.parse_elapsed(elapsed_time)
                    times[currentjobs[slurm_jobid]] = seconds_running

            return times

        def killJob(self, jobID: int) -> None:
            call_command(['scancel', self.getBatchSystemID(jobID)])

        def prepareSubmission(self,
                              cpu: int,
                              memory: int,
                              jobID: int,
                              command: str,
                              jobName: str,
                              job_environment: Optional[Dict[str, str]] = None) -> List[str]:
            return self.prepareSbatch(cpu, memory, jobID, jobName, job_environment) + ['--wrap={}'.format(command)]

        def prepareSubmissionArray(self,
                                   cpu: int,
                                   memory: int,
                                   arrayNumber: int,
                                   jobType: str,
                                   idx: int,
                                   arrayDirectory: str,
                                   job_environment: Optional[Dict[str, str]] = None) -> List[str]:
            jobID = "array" + str(arrayNumber)
            jobName = jobType
            return (
                self.prepareSbatch(cpu, memory, jobID, jobName, job_environment) + 
                ['--array=1-{}'.format(idx)] +
                ['--wrap="{}/in.$SLURM_ARRAY_TASK_ID"'.format(arrayDirectory)]
            )

        def submitJob(self, subLine: List[str]) -> int:
            try:
                # Slurm is not quite clever enough to follow the XDG spec on
                # its own. If the submission command sees e.g. XDG_RUNTIME_DIR
                # in our environment, it will send it along (especially with
                # --export=ALL), even though it makes a promise to the job that
                # Slurm isn't going to keep. It also has a tendency to create
                # /run/user/<uid> *at the start* of a job, but *not* keep it
                # around for the duration of the job.
                #
                # So we hide the whole XDG universe from Slurm before we make
                # the submission.
                # Might as well hide DBUS also.
                # This doesn't get us a trustworthy XDG session in Slurm, but
                # it does let us see the one Slurm tries to give us.
                no_session_environment = os.environ.copy()
                session_names = [n for n in no_session_environment.keys() if n.startswith('XDG_') or n.startswith('DBUS_')]
                for name in session_names:
                    del no_session_environment[name]

                output = call_command(subLine, env=no_session_environment)
                # sbatch prints a line like 'Submitted batch job 2954103'
                result = int(output.strip().split()[-1])
                logger.info("sbatch submitted job %s", result)
                subprocess.check_call([f"echo {result} >> $PWD/job_ids.txt"], shell=True)
                return result
            except OSError as e:
                logger.error(f"sbatch command failed with error: {e}")
                raise e

        def coalesce_job_exit_codes(self, batch_job_id_list: List[str]) -> List[Union[int, Tuple[int, Optional[BatchJobExitReason]], None]]:
            """
            Collect all job exit codes in a single call.
            :param batch_job_id_list: list of Job ID strings, where each string has the form
            "<job>[.<task>]".
            :return: list of job exit codes or exit code, exit reason pairs associated with the list of job IDs.
            """
            logger.log(TRACE, "Getting exit codes for slurm jobs: %s", batch_job_id_list)
            # Convert batch_job_id_list to list of integer job IDs.
            job_id_list = [int(id.split('.')[0]) for id in batch_job_id_list]
            status_dict = self._get_job_details(job_id_list)
            exit_codes: List[Union[int, Tuple[int, Optional[BatchJobExitReason]], None]] = []
            for _, status in status_dict.items():
                exit_codes.append(self._get_job_return_code(status))
            return exit_codes

        def getJobExitCode(self, slurm_job_id: str) -> Union[int, Tuple[int, Optional[BatchJobExitReason]], None]:
            """
            Get job exit code for given batch job ID.
            :param slurm_job_id: string of the form "<job>[.<task>]".
            :return: integer job exit code.
            """
            logger.log(TRACE, "Getting exit code for slurm job: %s", slurm_job_id)
            # Convert slurm_job_id to an integer job ID.
            job_id = int(slurm_job_id.split('.')[0])
            status_dict = self._get_job_details([job_id])
            status = status_dict[job_id]
            exit_status = self._get_job_return_code(status)
            if exit_status is None:
                return None

            exit_code, exit_reason = exit_status
            logger.info("Exit code for %s job is: %s, %s", slurm_job_id, str(exit_code), exit_reason)
            if exit_reason == BatchJobExitReason.MEMLIMIT:
                # Retry job with 2x memory if it was killed because of memory
                jobID = self._getJobID(slurm_job_id)
                exit_code = self._customRetry(jobID, slurm_job_id)
            return exit_code
        
        def _getJobID(self, slurm_job_id: str) -> int:
            """Get toil job ID from the slurm job ID."""
            job_ids_dict = {str(slurm_job[0]): int(toil_job) for toil_job, slurm_job in self.batchJobIDs.items()}
            if slurm_job_id not in job_ids_dict:
                raise RuntimeError(f"Unknown slurmJobID: {slurm_job_id}.\nTracked jobs: {job_ids_dict}")
            return job_ids_dict[slurm_job_id]
        
        def _customRetry(self, jobID: int, slurm_job_id: str) -> int:
            """Increase the job memory 2x and retry, when it's killed by memlimit problems."""
            try:
                jobNode = self.boss.Id2Node[jobID]
            except KeyError:
                logger.error("Can't resource retry %s, jobNode not found", jobID)
                return 1

            job_retries = self.boss.resourceRetryCount[jobID]
            if job_retries < OUT_OF_MEM_RETRIES:
                jobNode.jobName = (jobNode.jobName or "") + " OOM resource retry " + str(job_retries)
                memory = jobNode.memory * (job_retries + 1) * 2 if jobNode.memory < MAX_MEMORY else MAX_MEMORY

                sbatch_line = self.prepareSubmission(
                    jobNode.cores, memory, jobID, jobNode.command, jobNode.jobName
                )
                logger.debug("Running %r", sbatch_line)
                new_slurm_job_id = self.boss.with_retries(self.submitJob, sbatch_line)
                self.batchJobIDs[jobID] = (new_slurm_job_id, None)
                self.boss.resourceRetryCount[jobID] += 1
                logger.info(
                    "Detected job %s killed by SLURM, attempting retry with 2x memory: %s",
                    slurm_job_id, new_slurm_job_id
                )
                logger.info(
                    "Issued job %s with job batch system ID: "
                    "%s and cores: %s, disk: %s, and memory: %s",
                    jobNode, str(new_slurm_job_id), int(jobNode.cores),
                    bytes2human(jobNode.disk), bytes2human(memory)
                )
                with self.runningJobsLock:
                    self.runningJobs.add(jobID)
            else:
                logger.error("Can't retry job %s for memlimit more than twice")
                return 1
            return None
            
        def _get_job_details(self, job_id_list: List[int]) -> Dict[int, Tuple[Optional[str], Optional[int]]]:
            """
            Helper function for `getJobExitCode` and `coalesce_job_exit_codes`.
            Fetch job details from Slurm's accounting system or job control system.
            :param job_id_list: list of integer Job IDs.
            :return: dict of job statuses, where key is the integer job ID, and value is a tuple
            containing the job's state and exit code.
            """
            try:
                status_dict = self._getJobDetailsFromSacct(job_id_list)
            except CalledProcessErrorStderr:
                # no accounting system or some other error
                status_dict = self._getJobDetailsFromScontrol(job_id_list)
            return status_dict
        
        
        def _get_job_return_code(self, status: Tuple[Optional[str], Optional[int]]) -> Union[int, Tuple[int, Optional[BatchJobExitReason]], None]:
            """
            Given a Slurm return code, status pair, summarize them into a Toil return code, exit reason pair.
            The return code may have already been OR'd with the 128-offset
            Slurm-reported signal.
            Slurm will report return codes of 0 even if jobs time out instead
            of succeeding:
            
                2093597|TIMEOUT|0:0
                2093597.batch|CANCELLED|0:15
            
            So we guarantee here that, if the Slurm status string is not a
            successful one as defined in
            <https://slurm.schedmd.com/squeue.html#SECTION_JOB-STATE-CODES>, we
            will not return a successful return code.
            Helper function for `getJobExitCode` and `coalesce_job_exit_codes`.
            :param status: tuple containing the job's state and it's return code from Slurm.
            :return: the job's return code for Toil if it's completed, otherwise None.
            """
            state, rc = status

            if state not in TERMINAL_STATES:
                # Don't treat the job as exited yet
                return None

            exit_reason = TERMINAL_STATES[state]

            if exit_reason == BatchJobExitReason.FINISHED:
                # The only state that should produce a 0 ever is COMPLETED. So
                # if the job is COMPLETED and the exit reason is thus FINISHED,
                # pass along the code it has.
                return (rc, exit_reason)  # type: ignore[return-value] # mypy doesn't understand enums well

            if rc == 0:
                # The job claims to be in a state other than COMPLETED, but
                # also to have not encountered a problem. Say the exit status
                # is unavailable.
                return (EXIT_STATUS_UNAVAILABLE_VALUE, exit_reason)

            # If the code is nonzero, pass it along.
            return (rc, exit_reason)  # type: ignore[return-value] # mypy doesn't understand enums well

        def _canonicalize_state(self, state: str) -> str:
            """
            Turn a state string form SLURM into just the state token like "CANCELED".
            """

            # Slurm will sometimes send something like "CANCELED by 30065" in
            # the state column for some reason.

            state_token = state

            if " " in state_token:
                state_token = state.split(" ", 1)[0]

            if state_token not in TERMINAL_STATES and state_token not in NONTERMINAL_STATES:
                raise RuntimeError("Toil job in unimplemented Slurm state " + state)

            return state_token

    
        def _getJobDetailsFromSacct(self, job_id_list: List[int]) -> Dict[int, Tuple[Optional[str], Optional[int]]]:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `sacct`.
            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value is a tuple
            containing the job's state and exit code.
            """
            job_ids = ",".join(str(id) for id in job_id_list)
            args = ['sacct',
                    '-n',  # no header
                    '-j', job_ids,  # job
                    '--format', 'JobID,State,ExitCode',  # specify output columns
                    '-P',  # separate columns with pipes
                    '-S', '1970-01-01']  # override start time limit
            stdout = call_command(args, quiet=True)
            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `sacct`.
            job_statuses: Dict[int, Tuple[Optional[str], Optional[int]]] = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None)
        
            for line in stdout.split('\n'):
                values = line.strip().split('|')
                if len(values) < 3:
                    continue
                state: str
                job_id_raw, state, exitcode = values
                state = self._canonicalize_state(state)
                logger.log(TRACE, "%s state of job %s is %s", args[0], job_id_raw, state)
                # JobIDRaw is in the form JobID[.JobStep]; we're not interested in job steps.
                job_id_parts = job_id_raw.split(".")
                if len(job_id_parts) > 1:
                    continue
                job_id = job_id_parts[0]
                status: int
                signal: int
                status, signal = (int(n) for n in exitcode.split(':'))
                if signal > 0:
                    # A non-zero signal may indicate e.g. an out-of-memory killed job
                    status = 128 + signal
                logger.log(TRACE, "%s exit code of job %s is %s, return status %d",
                             args[0], job_id, exitcode, status)
                job_statuses[job_id] = state, status
            logger.log(TRACE, "%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        def _getJobDetailsFromScontrol(self, job_id_list: List[int]) -> Dict[int, Tuple[Optional[str], Optional[int]]]:
            """
            Get SLURM job exit codes for the jobs in `job_id_list` by running `scontrol`.
            :param job_id_list: list of integer batch job IDs.
            :return: dict of job statuses, where key is the job-id, and value is a tuple
            containing the job's state and exit code.
            """
            args = ['scontrol',
                    'show',
                    'job']
            
            # `scontrol` can only return information about a single job,
            # or all the jobs it knows about.
            if len(job_id_list) == 1:
                args.append(str(job_id_list[0]))

            stdout = call_command(args, quiet=True)
            
            job_records = None
            if isinstance(stdout, str):
                job_records = stdout.strip().split('\n\n')
            elif isinstance(stdout, bytes):
                job_records = stdout.decode('utf-8').strip().split('\n\n')

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `scontrol`.
            job_statuses: Dict[int, Tuple[Optional[str], Optional[int]]] = {}
            job_id: Optional[int]
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None)

            # `scontrol` will report "No jobs in the system", if there are no jobs in the system,
            # and if no job-id was passed as argument to `scontrol`.
            if len(job_records) > 0 and job_records[0] == "No jobs in the system":
                return job_statuses

            for record in job_records:
                job: Dict[str, str] = {}
                job_id = None
                for line in record.splitlines():
                    for item in line.split():
                        # Output is in the form of many key=value pairs, multiple pairs on each line
                        # and multiple lines in the output. Each pair is pulled out of each line and
                        # added to a dictionary.
                        # Note: In some cases, the value itself may contain white-space. So, if we find
                        # a key without a value, we consider that key part of the previous value.
                        bits = item.split('=', 1)
                        if len(bits) == 1:
                            job[key] += ' ' + bits[0]  # type: ignore[has-type]  # we depend on the previous iteration to populate key
                        else:
                            key = bits[0]
                            job[key] = bits[1]
                    # The first line of the record contains the JobId. Stop processing the remainder
                    # of this record, if we're not interested in this job.
                    job_id = job['JobId']
                    if job_id not in job_id_list:
                        logger.log(TRACE, "%s job %s is not in the list", args[0], job_id)
                        break
                if job_id is None or job_id not in job_id_list:
                    continue
                state = job['JobState']
                state = self._canonicalize_state(state)
                logger.log(TRACE, "%s state of job %s is %s", args[0], job_id, state)
                try:
                    exitcode = job['ExitCode']
                    if exitcode is not None:
                        status, signal = (int(n) for n in exitcode.split(':'))
                        if signal > 0:
                            # A non-zero signal may indicate e.g. an out-of-memory killed job
                            status = 128 + signal
                        logger.log(TRACE, "%s exit code of job %s is %s, return status %d",
                                     args[0], job_id, exitcode, status)
                        rc = status
                    else:
                        rc = None
                except KeyError:
                    rc = None
                job_statuses[job_id] = (state, rc)
            logger.log(TRACE, "%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        ###
        ### Implementation-specific helper methods
        ###
        def prepareSbatch(self,
                          cpu: int,
                          mem: int,
                          jobID: int,
                          jobName: str,
                          job_environment: Optional[Dict[str, str]] = None) -> List[str]:

            """
            Returns the sbatch command line to run to queue the job.
            """
            
            # Start by naming the job
            sbatch_line = ['sbatch', '-J', f'toil_job_{jobID}_{jobName}']

            # Make sure the job gets a signal before it disappears so that e.g.
            # container cleanup finally blocks can run. Ask for SIGINT so we
            # can get the default Python KeyboardInterrupt which third-party
            # code is likely to plan for. Make sure to send it to the batch
            # shell process with "B:", not to all the srun steps it launches
            # (because there shouldn't be any). We cunningly replaced the batch
            # shell process with the Toil worker process, so Toil should be
            # able to get the signal.
            #
            # TODO: Add a way to detect when the job failed because it
            # responded to this signal and use the right exit reason for it.
            sbatch_line.append("--signal=B:INT@30")

            environment = {}
            environment.update(self.boss.environment)
            if job_environment:
                environment.update(job_environment)
                
            # "Native extensions" for SLURM (see DRMAA or SAGA)
            nativeConfig = os.getenv('TOIL_SLURM_ARGS')

            # --export=[ALL,]<environment_toil_variables>
            set_exports = "--export=ALL"

            if nativeConfig is not None:
                logger.debug("Native SLURM options appended to sbatch from TOIL_SLURM_ARGS env. variable: %s", nativeConfig)

                for arg in nativeConfig.split():
                    if arg.startswith("--mem") or arg.startswith("--cpus-per-task"):
                        raise ValueError(f"Some resource arguments are incompatible: {nativeConfig}")
                    # repleace default behaviour by the one stated at TOIL_SLURM_ARGS
                    if arg.startswith("--export"):
                        set_exports = arg
                sbatch_line.extend(nativeConfig.split())

            if environment:
                argList = []

                for k, v in environment.items():
                    quoted_value = quote(os.environ[k] if v is None else v)
                    argList.append(f'{k}={quoted_value}')
                
                set_exports += ',' + ','.join(argList)

            # add --export to the sbatch
            sbatch_line.append(set_exports)
            
            parallel_env = os.getenv('TOIL_SLURM_PE')
            if cpu and cpu > 1 and parallel_env:
                sbatch_line.append(f'--partition={parallel_env}')

            if mem is not None and self.boss.config.allocate_mem:  # type: ignore[attr-defined]
                # memory passed in is in bytes, but slurm expects megabytes
                sbatch_line.append(f'--mem={math.ceil(mem / 2 ** 20)}')
            if cpu is not None:
                sbatch_line.append(f'--cpus-per-task={math.ceil(cpu)}')

            stdoutfile: str = self.boss.formatStdOutErrPath(jobID, '%j', 'out')
            stderrfile: str = self.boss.formatStdOutErrPath(jobID, '%j', 'err')
            sbatch_line.extend(['-o', stdoutfile, '-e', stderrfile])
            
            return sbatch_line
        
        def prepareJobScript(self, command: str, idx: int, arrayDirectory: str):
            # Creates a sh script for a job
            # Define the script file path
            script_path = os.path.join(arrayDirectory, "in.{}".format(idx))

            # Write the command into the script
            with open(script_path, 'w') as script_file:
                script_file.write("#!/bin/bash\n")
                script_file.write(command + "\n")

            os.chmod(script_path, 0o755)

        def parse_elapsed(self, elapsed: str) -> int:
            # slurm returns elapsed time in days-hours:minutes:seconds format
            # Sometimes it will only return minutes:seconds, so days may be omitted
            # For ease of calculating, we'll make sure all the delimeters are ':'
            # Then reverse the list so that we're always counting up from seconds -> minutes -> hours -> days
            total_seconds = 0
            try:
                elapsed_split: List[str] = elapsed.replace('-', ':').split(':')
                elapsed_split.reverse()
                seconds_per_unit = [1, 60, 3600, 86400]
                for index, multiplier in enumerate(seconds_per_unit):
                    if index < len(elapsed_split):
                        total_seconds += multiplier * int(elapsed_split[index])
            except ValueError:
                pass  # slurm may return INVALID instead of a time
            return total_seconds

    """
    The interface for SLURM
    """

    @classmethod
    def getWaitDuration(cls):
        # Extract the slurm batchsystem config for the appropriate value
        wait_duration_seconds = 1
        lines = call_command(['scontrol', 'show', 'config']).split('\n')
        time_value_list = []
        for line in lines:
            values = line.split()
            if len(values) > 0 and (values[0] == "SchedulerTimeSlice" or values[0] == "AcctGatherNodeFreq"):
                time_name = values[values.index('=')+1:][1]
                time_value = int(values[values.index('=')+1:][0])
                if time_name == 'min':
                    time_value *= 60
                # Add a 20% ceiling on the wait duration relative to the scheduler update duration
                time_value_list.append(math.ceil(time_value*1.2))
        return max(time_value_list)
