"""Functions to analyze compiled binaries"""

import os
import subprocess
import shutil
from utils.misc import MAX_STD_STR_LEN, get_symlink_binds
from utils.logs import MPLogger
from parsing.container_info import get_container_platform, CONTAINER_PLATFORMS, get_container_path, MOUNTING_DIR


# The logger to use
LOGGER = MPLogger()


def run_analysis_cmd(paths, binary_fp, analyzer_info, container_platform):
    """Sends the input through an analyzer to analyze

    Will move (not copy) the binary_fp into path/temp if it doesn't exist there already.

    Will create a cfg output file at path/temp/binary_fp_basename.cfg

    Will remove any extra intermediary files

    Args:
        paths (Dict[str, str]): the dictionary of file paths. See main.main_codeforces() for more info
        binary_fp (Union[str, List[str]]): the file path to the binary to analyze, or multiple paths if there are multiple
            files associated with this one source code file. If there are multiple paths, then the 'main' file to analyze
            is expected to be the first element
        analyzer_info (Dict): the analyzer information to use. Should contain 'container' and 'analysis_cmd' keys
        container_platform (str): the string container platform to use

    Returns:
        str: the filepath to the analysis cfg file
    """
    binary_fp = binary_fp if isinstance(binary_fp, str) else binary_fp[0]
    container_platform = get_container_platform(container_platform)

    # Move the binary to the temp directory if needed
    if os.path.dirname(binary_fp) != paths['temp']:
        new_fp = os.path.join(paths['temp'], os.path.basename(binary_fp))
        shutil.copy(binary_fp, new_fp)
        binary_fp = new_fp
    output_cfg_path = os.path.join(paths['temp'], os.path.basename(binary_fp) + ".cfg")

    # Generate the analysis command
    container_dict = CONTAINER_PLATFORMS[container_platform]
    analysis_cmd = analyzer_info['analysis_cmd'].format(binary_basename=os.path.basename(binary_fp), 
                                                        analyzer_output_basename=os.path.basename(output_cfg_path))
    bind_cmd = container_dict['bind_command'].format(host_path=paths['temp'], container_path=MOUNTING_DIR)
    extra_args = get_symlink_binds(container_dict['bind_command'], paths['temp'])
    command = container_dict['execution_command'].format(dir_bind=bind_cmd, extra_args=extra_args, command=analysis_cmd,
                                                         container=get_container_path(paths, analyzer_info, container_platform))

    # Execute the command
    LOGGER.debug("Executing analysis command: '%s'" % command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    
    return output_cfg_path, (stdout.decode('utf-8')[:MAX_STD_STR_LEN], stderr.decode('utf-8')[:MAX_STD_STR_LEN])
