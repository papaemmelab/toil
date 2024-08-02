import logging
import os
import random
import shutil
import subprocess
import sys
import typing
import datetime

from typing import Iterator, Union, List, Optional

logger = logging.getLogger(__name__)


def printq(msg: str, quiet: bool) -> None:
    if not quiet:
        print(msg)


def truncExpBackoff() -> Iterator[float]:
    # as recommended here https://forums.aws.amazon.com/thread.jspa?messageID=406788#406788
    # and here https://cloud.google.com/storage/docs/xml-api/reference-status
    yield 0
    t = 1
    while t < 1024:
        # google suggests this dither
        yield t + random.random()
        t *= 2
    while True:
        yield t


class CalledProcessErrorStderr(subprocess.CalledProcessError):
    """Version of CalledProcessError that include stderr in the error message if it is set"""

    def __str__(self) -> str:
        if (self.returncode < 0) or (self.stderr is None):
            return str(super())
        else:
            err = self.stderr if isinstance(self.stderr, str) else self.stderr.decode("ascii", errors="replace")
            return "Command '%s' exit status %d: %s" % (self.cmd, self.returncode, err)


def call_command(cmd: List[str], *args: str, input: Optional[str] = None, timeout: Optional[float] = None,
                useCLocale: bool = True, env: Optional[typing.Dict[str, str]] = None, quiet: Optional[bool] = False) -> Union[str, bytes]:
    """
    Simplified calling of external commands.
    If the process fails, CalledProcessErrorStderr is raised.
    The captured stderr is always printed, regardless of
    if an exception occurs, so it can be logged.
    Always logs the command at debug log level.
    :param quiet: If True, do not log the command output. If False (the
           default), do log the command output at debug log level.
    :param useCLocale: If True, C locale is forced, to prevent failures that
           can occur in some batch systems when using UTF-8 locale.
    :returns: Command standard output, decoded as utf-8.
    """
    # NOTE: Interface MUST be kept in sync with call_sacct and call_scontrol in
    # test_slurm.py, which monkey-patch this!

    # using non-C locales can cause GridEngine commands, maybe other to
    # generate errors
    if useCLocale:
        env = dict(os.environ) if env is None else dict(env)  # copy since modifying
        env["LANGUAGE"] = env["LC_ALL"] = "C"

    logger.debug("run command: {}".format(" ".join(cmd)))
    start_time = datetime.datetime.now()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding='utf-8', errors="replace", env=env)
    stdout, stderr = proc.communicate(input=input, timeout=timeout)
    end_time = datetime.datetime.now()
    runtime = (end_time - start_time).total_seconds()
    sys.stderr.write(stderr)
    if proc.returncode != 0:
        logger.debug("command failed in {}s: {}: {}".format(runtime, " ".join(cmd), stderr.rstrip()))
        raise CalledProcessErrorStderr(proc.returncode, cmd, output=stdout, stderr=stderr)
    logger.debug("command succeeded in {}s: {}: {}".format(runtime, " ".join(cmd), (': ' + stdout.rstrip()) if not quiet else ''))
    return stdout
