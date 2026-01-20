"""
Logging utilities
"""

import logging
import math
import os
import sys
import traceback
import socket
from utils.misc import CriticalError


LOGGER_NAME = 'logger-%d'


def init_logging(log_dir, task_id, exec_uid=None, file_name=None, with_stdout=False, stdout_level=logging.DEBUG, 
    separate_errors=False, max_task_id=None):
    """
    Initializes logging to output to a file. Assumes log_dir already exists

    :param log_dir: the directory for log files (assumed to have already been created)
    :param task_id: the task_id of this CAP instance
    :param exec_uid: the execution uid of this CAP process
    :param file_name: the name for the log file. Log files will be of format: [TIME]_[NAME]_[debug/error].log. This
        determines what will be in the 'NAME' slot. If None, then the name will be [task_id]
    :param with_stdout: if True, then will also log output to stdout
    :param stdout_level: the logging level to use for stdout. Only used if with_stdout is True
    :param separate_errors: if True, then a separate file will be created to store only the errors that occured (the
        errors are still stored in the debug file, this just makes it easier to find the errors in large debug files)
    :param max_task_id: if passed, then the maximum task_id value for a large job. Used to put leading zeros in front
        of the task_id in the filename so log files will always show up sorted by the file system
    """
    formatter = logging.Formatter('TASK-{} %(levelname)s (%(asctime)s): %(message)s'.format(task_id))
    format_str = ('%%s%%0%dd' % max(1, int(math.log10(max_task_id)) + 1)) if max_task_id is not None else '%s%d'
    file_name = file_name if file_name is not None else (format_str % ((exec_uid + "_") if exec_uid is not None else '', task_id))

    logger = logging.getLogger(LOGGER_NAME % task_id)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(log_dir, '%s_debug.log' % file_name), 'w')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if separate_errors:
        eh = logging.FileHandler(os.path.join(log_dir, '%s_error.log' % file_name), 'w')
        eh.setFormatter(formatter)
        eh.setLevel(logging.ERROR)
        logger.addHandler(eh)

    if with_stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(stdout_level)
        logger.addHandler(console)

    logger.info("Initialized Logging for task number: %d, running on machine: %s" % (task_id, socket.gethostname()))


class ExceptionLogger:
    def __init__(self, logger, handle=False, message=None):
        self.logger = logger
        self.handle = handle
        self.message = message

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is not None:
            if self.message is not None:
                self.logger.error(self.message + "\n" + str(traceback.format_exc()))
            else:
                self.logger.critical(traceback.format_exc())
            return self.handle and exc_type not in [KeyboardInterrupt, SystemExit, CriticalError]


class MPLogger:
    """A logger, if task_id is None, then it will be checked when first log is attempted"""
    def __init__(self, task_id=None):
        self.task_id = task_id
    
    def get_logger(self):
        if self.task_id is None:
            from __main__ import LOGGER
            self.task_id = LOGGER.task_id
        return logging.getLogger(LOGGER_NAME % self.task_id)
    
    def log(self, level, msg, *args):
        self.get_logger().log(level, msg, *args)
    
    def debug(self, msg, *args):
        self.get_logger().debug(msg, *args)
    
    def info(self, msg, *args):
        self.get_logger().info(msg, *args)
    
    def warn(self, msg, *args):
        self.get_logger().warn(msg, *args)
    
    def warning(self, msg, *args):
        self.get_logger().warn(msg, *args)
    
    def err(self, msg, *args):
        self.get_logger().error(msg, *args)
    
    def error(self, msg, *args):
        self.get_logger().error(msg, *args)
    
    def crit(self, msg, *args):
        self.get_logger().critical(msg, *args)
    
    def critical(self, msg, *args):
        self.get_logger().critical(msg, *args)
