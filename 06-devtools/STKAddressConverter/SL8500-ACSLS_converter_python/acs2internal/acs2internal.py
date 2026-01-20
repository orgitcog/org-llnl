#!/usr/bin/env python

# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Huy Le <le35@llnl.gov>
# LLNL-CODE-734258
#
# All rights reserved.
# This file is part of STK Address Converter. For details, see
# https://github.com/LLNL/STKAddressConverter. Licensed under the
# Apache License, Version 2.0 (the “Licensee”); you may not use
# this file except in compliance with the License. You may
# obtain a copy of the License at:
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the license.
##
##
# Converts ACSLS Drive Address to Internal Address.
# Converts Internal Address to ACSLS Drive Address.
#
# Usage: python acs2internal.py -d 1,10,1,4
# Example: 1,10,1,4 to 3,3,-1,1,1
# STDOUT: 3,3,-1,1,1
#
# Usage: python acs2internal.py -i 3,3,-1,1,1
# Example: 3,3,-1,1,1 to 1,10,1,4
# STDOUT: 1,10,1,4
#
# --debug provides additional output.
#

from argparse import ArgumentParser, RawTextHelpFormatter
from subprocess import Popen, PIPE, CalledProcessError
from sys import exit as sys_exit

# Converts ACSLS address to internal address or vice versa.
# --debug flag prints the stdout and stderr.
PARSER = ArgumentParser(description="Converts ACSLS address to " + \
                                    "internal address or vice " + \
                                    "versa.",
                        epilog="Example:\npython acs2internal.py " + \
                               "-d 1,10,1,4\n" + \
                               "python acs2internal.py " + \
                               "-i 3,3,-1,1,1",
                        formatter_class=RawTextHelpFormatter)
PARSER.add_argument("--debug", help="Print stdout and stderr messages.",
                    action="store_true")
PARSER.add_argument("-d", "--driveaddr", help="Input ACS / HLI-PRC address.",
                    required=False)
PARSER.add_argument("-i", "--intaddr", help="Input Internal Address.",
                    required=False)
ARGS = PARSER.parse_args()

# Global variable for the ACS drive id table.
ACSLS_DRIVE_IDS = [[12, 8, 4, 0],
                   [13, 9, 5, 1],
                   [14, 10, 6, 2],
                   [15, 11, 7, 3]]

def main():
    """Main function that checks arguments and uses helper functions
       to translate acsls or internal address accordingly.
    """
    # One of these address flags must be on for translation to happen.
    if ARGS.driveaddr and ARGS.intaddr:
        print("--driveaddr and --intaddr are both enabled.\n" + \
              "Don't use both flags at the same time.")
        sys_exit(10)
    elif not ARGS.driveaddr and not ARGS.intaddr:
        print("--driveaddr and --intaddr are both not enabled.\n" + \
              "Use at least either flag.")
        sys_exit(11)

    if ARGS.driveaddr:
        acsls_addr_to_internal_addr()

    if ARGS.intaddr:
        internal_addr_to_acsls_addr()

def acsls_addr_to_internal_addr(acs_address=None):
    """Converts ACSLS address to Internal address and prints.

    Args:
        acs_address: String of the acs drive address like 1,10,1,4.

    Returns:
        internal_address: String of the internal address
                          translated from the acs_address

    Example: 1,10,1,4 to 3,3,-1,1,1

    Breaking down the ACSLS address:
        1 is used for tape drives.
        lsm = 10
        1
        drive_id = 4

    Breaking down the Internal address:
        library = 3
        rail = 3
        column = -1
        side = 1
        row = 1
    """
    if (not acs_address):
        acs_address = ARGS.driveaddr
    split_acs_address = acs_address.split(",")

    # Check for correct format by length of the array split by ,
    if (len(split_acs_address) != 4):
        print("ACSLS address is in an incorrect format. Example: 1,10,1,4")
        sys_exit(1)

    if (ARGS.debug):
        print("ACSLS Address: " + acs_address)

    # Extra integer validation
    try:
        lsm = int(split_acs_address[1])
        drive_id = int(split_acs_address[3])
    except ValueError:
        print("Each number in the address must be a valid integer. " + \
              "Exiting...")
        sys_exit(1)

    # We find the array_row and array_column where the drive_id matches a cell.
    for i, row in enumerate(ACSLS_DRIVE_IDS):
        for j, cell in enumerate(row):
            if (cell == drive_id):
                array_row = i
                array_column = j

    row = array_row + 1

    # columns can be one of four values:
    # 2 1 -1 -2
    # We translate the column index to the rolumn.
    # And we multiply the -1 to get one of the four values.
    if array_column >= 2:
        column = array_column - 1
    elif array_column < 2:
        column = array_column - 2

    column *= -1

    # LSM equation according to internal_addr_to_acsls_addr():
    # lsm = ((4 * (library - 1)) + rail) - 1
    library = (lsm / 4) + 1
    rail = (lsm % 4) + 1

    # side is always 1 because tape drive bays only exist on side 1 of the
    # library
    side = 1

    internal_address = str(library) + "," + \
                       str(rail) + "," + \
                       str(column) + "," + \
                       str(side) + "," + \
                       str(row)

    if (ARGS.debug):
        print("Internal Address: " + internal_address)

    print(internal_address)

    return (internal_address.replace(" ", ""))

def internal_addr_to_acsls_addr(internal_address=None):
    """Converts Internal address to ACSLS address and prints.

    Args:
        internal_address: String of the internal address like 3,3,-1,1,1.

    Returns:
        acs_address: String of the acs address
                     translated from the internal_address

    Example: 3,3,-1,1,1 to 1,10,1,4

    Breaking down the Internal address:
        library = 3
        rail = 3
        column = -1
        side = 1
        row = 1

    Breaking down the ACSLS address:
        1 is used for tape drives.
        lsm = 10
        1
        drive_id = 4
    """
    if (not internal_address):
        internal_address = ARGS.intaddr
    split_internal_address = internal_address.split(",")

    # Check for correct format by length of the array split by ,
    if (len(split_internal_address) != 5):
        print("Internal address is in an incorrect format. " + \
              "Example: 3,3,-1,1,1")
        sys_exit(1)

    if (ARGS.debug):
        print("Internal Address: " + internal_address)

    # Grab the input arguments.
    try:
        library = int(split_internal_address[0])
        rail = int(split_internal_address[1])
        column = int(split_internal_address[2])
        side = int(split_internal_address[3])
        row = int(split_internal_address[4])
    except ValueError:
        print("Each number in the address must be a valid integer. " + \
              "Exiting...\n")
        sys_exit(1)

    # Validate the inputs.
    if ((library < 1) or \
        ((rail < 1) or (rail > 4)) or \
        ((column < -2) or (column > 2) or (column == 0)) or \
        (side != 1) or \
        ((row < 1) or (row > 4))):
        sys_exit(1)

    # Calculate the ACSLS LSM ID.
    lsm = ((4 * (library - 1)) + rail) - 1

    # Convert LT05 row to array row.
    array_row = row - 1

    # Convert the column to array column.
    # columns can be one of four values:
    # 2 1 -1 -2
    # and we need to translate them into a valid array rolumn to index
    # into the array defined at the top of the function. Thus. multiplying by
    # -1 and adding either 2 or 1 if negative or positive (respectively), we
    # translate the values into:
    # 0 1 2 3
    column *= -1
    if (column < 0):
        array_column = column + 2
    elif (column > 0):
        array_column = column + 1
    else:
        # Column is equal to zero. This cannot happen. Abort.
        print("Internal error: internal column representation is incorrect.\n")

    # Now that we have a row and column for the drive ID array, retrieve the ID.
    drive_id = ACSLS_DRIVE_IDS[array_row][array_column]

    acsls_address = "1," + str(lsm) + ",1," + str(drive_id)

    if (ARGS.debug):
        print("ACSLS Address: " + acsls_address)

    print(acsls_address)

    return acsls_address

def run_command(name, return_code, command):
    """Runs arbitrary command and exits with return_code if failure.

    Args:
        name: Name of the command for identification purposes.
        return_code: An integer to exit if command fails.
        command: An array with the components of the command.

    Raises:
        OSError or CalledProcessError: An error occurred running the
        command.

    Returns:
        output: A string with the STDOUT.
    """
    try:
        process = Popen(command,
                        stdout=PIPE,
                        stderr=PIPE)
    except (OSError, CalledProcessError) as err:
        print(name + " runtime error: " + str(err))
        sys_exit(return_code)

    # Check standard output and error of the command and print in debug mode.
    output, error = process.communicate()

    if (ARGS.debug):
        print(output)
        print(error)

    if (error):
        print(" ".join(command))
        print(error)
        sys_exit(return_code)

    return output

if __name__ == "__main__":
    main()
