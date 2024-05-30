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

from __future__ import absolute_import
from __future__ import division
from builtins import str
from past.utils import old_div
import logging
import os
from pipes import quote
from toil import subprocess
import time
import math

# Python 3 compatibility imports
from six.moves.queue import Empty, Queue
from six import iteritems

from toil.batchSystems import MemoryString
from toil.batchSystems.abstractGridEngineBatchSystem import AbstractGridEngineBatchSystem, BatchJobExitReason, EXIT_STATUS_UNAVAILABLE_VALUE
from toil.lib.misc import CalledProcessErrorStderr, call_command

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
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
NONTERMINAL_STATES = {
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

    class Worker(AbstractGridEngineBatchSystem.Worker):

        def getRunningJobIDs(self):
            # Should return a dictionary of Job IDs and number of seconds
            times = {}
            with self.runningJobsLock:
                currentjobs = dict((str(self.batchJobIDs[x][0]), x) for x in self.runningJobs)
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

        def killJob(self, jobID):
            call_command(['scancel', self.getBatchSystemID(jobID)])

        def prepareSubmission(self, cpu, memory, jobID, command, jobName):
            return self.prepareSbatch(cpu, memory, jobID, jobName) + ['--wrap={}'.format(command)]

        def submitJob(self, subLine):
            try:
                output = call_command(subLine)
                # sbatch prints a line like 'Submitted batch job 2954103'
                result = int(output.strip().split()[-1])
                logger.debug("sbatch submitted job %d", result)
                return result
            except OSError as e:
                logger.error("sbatch command failed")
                raise e

        def getJobExitCode(self, batchJobID):
            """
            Get job exit code for given batch job ID.
            :param batchJobID: string of the form "<job>[.<task>]".
            :return: integer job exit code.
            """
            logger.debug("Getting exit code for slurm job %d", int(batchJobID))

            job_id = int(batchJobID.split('.')[0])
            status_dict = self._get_job_details([job_id])
            status = status_dict[job_id]
            return self._get_job_return_code(status)

        def _get_job_details(self, job_id_list):
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
                status_dict = self._getJobDetailsFromScontrol(job_id_list)
            return status_dict

        def _get_job_return_code(self, status):
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
                return (rc, exit_reason)

            if rc == 0:
                # The job claims to be in a state other than COMPLETED, but
                # also to have not encountered a problem. Say the exit status
                # is unavailable.
                return (EXIT_STATUS_UNAVAILABLE_VALUE, exit_reason)

            # If the code is nonzero, pass it along.
            return (rc, exit_reason)

        def _canonicalize_state(self, state):
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

        def _getJobDetailsFromSacct(self, job_id_list):
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
                    '--format', 'JobIDRaw,State,ExitCode',  # specify output columns
                    '-P',  # separate columns with pipes
                    '-S', '1970-01-01']  # override start time limit
            stdout = call_command(args, quiet=True)

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `sacct`.
            job_statuses = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None)

            for line in stdout.splitlines():
                values = line.strip().split('|')
                if len(values) < 3:
                    continue
                job_id_raw, state, exitcode = values
                state = self._canonicalize_state(state)
                logger.debug("%s state of job %s is %s", args[0], job_id_raw, state)
                # JobIDRaw is in the form JobID[.JobStep]; we're not interested in job steps.
                job_id_parts = job_id_raw.split(".")
                if len(job_id_parts) > 1:
                    continue
                job_id = int(job_id_parts[0])
                status, signal = (int(n) for n in exitcode.split(':'))
                if signal > 0:
                    # A non-zero signal may indicate e.g. an out-of-memory killed job
                    status = 128 + signal
                logger.debug("%s exit code of job %d is %s, return status %d",
                             args[0], job_id, exitcode, status)
                job_statuses[job_id] = state, status
            logger.debug("%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        def _getJobDetailsFromScontrol(self, job_id_list):
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

            # Job records are separated by a blank line.
            if isinstance(stdout, str):
                job_records = stdout.strip().split('\n\n')
            elif isinstance(stdout, bytes):
                job_records = stdout.decode('utf-8').strip().split('\n\n')

            # Collect the job statuses in a dict; key is the job-id, value is a tuple containing
            # job state and exit status. Initialize dict before processing output of `scontrol`.
            job_statuses = {}
            for job_id in job_id_list:
                job_statuses[job_id] = (None, None)

            # `scontrol` will report "No jobs in the system", if there are no jobs in the system,
            # and if no job-id was passed as argument to `scontrol`.
            if len(job_records) > 0 and job_records[0] == "No jobs in the system":
                return job_statuses

            for record in job_records:
                job = {}
                for line in record.splitlines():
                    for item in line.split():
                        # Output is in the form of many key=value pairs, multiple pairs on each line
                        # and multiple lines in the output. Each pair is pulled out of each line and
                        # added to a dictionary.
                        # Note: In some cases, the value itself may contain white-space. So, if we find
                        # a key without a value, we consider that key part of the previous value.
                        bits = item.split('=', 1)
                        if len(bits) == 1:
                            job[key] += ' ' + bits[0]
                        else:
                            key = bits[0]
                            job[key] = bits[1]
                    # The first line of the record contains the JobId. Stop processing the remainder
                    # of this record, if we're not interested in this job.
                    job_id = int(job['JobId'])
                    if job_id not in job_id_list:
                        logger.debug("%s job %d is not in the list", args[0], job_id)
                        break
                if job_id not in job_id_list:
                    continue
                state = job['JobState']
                state = self._canonicalize_state(state)
                logger.debug("%s state of job %s is %s", args[0], job_id, state)
                try:
                    exitcode = job['ExitCode']
                    if exitcode is not None:
                        status, signal = (int(n) for n in exitcode.split(':'))
                        if signal > 0:
                            # A non-zero signal may indicate e.g. an out-of-memory killed job
                            status = 128 + signal
                        logger.debug("%s exit code of job %d is %s, return status %d",
                                     args[0], job_id, exitcode, status)
                        rc = status
                    else:
                        rc = None
                except KeyError:
                    rc = None
                job_statuses[job_id] = (state, rc)
            logger.debug("%s returning job statuses: %s", args[0], job_statuses)
            return job_statuses

        """
        Implementation-specific helper methods
        """

        def prepareSbatch(self, cpu, mem, jobID, jobName):
            #  Returns the sbatch command line before the script to run
            sbatch_line = ['sbatch', '-J', 'toil_job_{}_{}'.format(jobID, jobName)]

            if self.boss.environment:
                argList = []
                
                for k, v in self.boss.environment.items():
                    quoted_value = quote(os.environ[k] if v is None else v)
                    argList.append('{}={}'.format(k, quoted_value))
                    
                sbatch_line.append('--export=' + ','.join(argList))
            
            if mem is not None:
                # memory passed in is in bytes, but slurm expects megabytes
                sbatch_line.append('--mem={}'.format(old_div(int(mem), 2 ** 20)))
            if cpu is not None:
                sbatch_line.append('--cpus-per-task={}'.format(int(math.ceil(cpu))))

            stdoutfile = self.boss.formatStdOutErrPath(jobID, 'slurm', '%j', 'std_output')
            stderrfile = self.boss.formatStdOutErrPath(jobID, 'slurm', '%j', 'std_error')
            sbatch_line.extend(['-o', stdoutfile, '-e', stderrfile])

            # "Native extensions" for SLURM (see DRMAA or SAGA)
            nativeConfig = os.getenv('TOIL_SLURM_ARGS')
            if nativeConfig is not None:
                logger.debug("Native SLURM options appended to sbatch from TOIL_SLURM_ARGS env. variable: {}".format(nativeConfig))
                if ("--mem" in nativeConfig) or ("--cpus-per-task" in nativeConfig):
                    raise ValueError("Some resource arguments are incompatible: {}".format(nativeConfig))

                sbatch_line.extend(nativeConfig.split())

            return sbatch_line

        def parse_elapsed(self, elapsed):
            # slurm returns elapsed time in days-hours:minutes:seconds format
            # Sometimes it will only return minutes:seconds, so days may be omitted
            # For ease of calculating, we'll make sure all the delimeters are ':'
            # Then reverse the list so that we're always counting up from seconds -> minutes -> hours -> days
            total_seconds = 0
            try:
                elapsed = elapsed.replace('-', ':').split(':')
                elapsed.reverse()
                seconds_per_unit = [1, 60, 3600, 86400]
                for index, multiplier in enumerate(seconds_per_unit):
                    if index < len(elapsed):
                        total_seconds += multiplier * int(elapsed[index])
            except ValueError:
                pass  # slurm may return INVALID instead of a time
            return total_seconds

    """
    The interface for SLURM
    """

    @classmethod
    def getWaitDuration(cls):
        return 1

    @classmethod
    def obtainSystemConstants(cls):
        # sinfo -Ne --format '%m,%c'
        # sinfo arguments:
        # -N for node-oriented
        # -h for no header
        # -e for exact values (e.g. don't return 32+)
        # --format to get memory, cpu
        max_cpu = 0
        max_mem = MemoryString('0')
        lines = subprocess.check_output(['sinfo', '-Nhe', '--format', '%m %c']).decode('utf-8').split('\n')
        for line in lines:
            values = line.split()
            if len(values) < 2:
                continue
            mem, cpu = values
            max_cpu = max(max_cpu, int(cpu))
            max_mem = max(max_mem, MemoryString(mem + 'M'))
        if max_cpu == 0 or max_mem.byteVal() == 0:
            RuntimeError('sinfo did not return memory or cpu info')
        return max_cpu, max_mem
