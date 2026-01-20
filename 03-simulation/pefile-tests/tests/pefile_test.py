# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2004-2016 Ero Carrera
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import difflib
from hashlib import sha256
import unittest

from io import open
import pytest

import pefile

# do not truncate outputs
pefile.MAX_SECTIONS = 128000


here = os.path.abspath(__file__)
test_dir = os.path.dirname(here)
REGRESSION_TESTS_DIR = os.path.join(test_dir, "data")
POCS_TESTS_DIR = os.path.join(test_dir, "corkami/pocs")
LIEF_TESTS_DIR = os.path.join(test_dir, "lief-samples-PE")


def _load_test_files():
    """Yield all the test files"""

    not_pes = ".dmp", ".ABOUT", "empty_file",
    for dirpath, _dirname, filenames in os.walk(REGRESSION_TESTS_DIR):
        for filename in filenames:
            if not filename.endswith(not_pes):
                yield os.path.join(dirpath, filename)

    for dirpath, _dirname, filenames in os.walk(POCS_TESTS_DIR):
        for filename in filenames:
            if not filename.endswith(not_pes):
                yield os.path.join(dirpath, filename)

    for dirpath, _dirname, filenames in os.walk(LIEF_TESTS_DIR):
        for filename in filenames:
            if not filename.endswith(not_pes):
                yield os.path.join(dirpath, filename)


@pytest.mark.parametrize(
    "pe_filename",
    list(_load_test_files()),
)
def test_pe_image_regression_test(pe_filename, REGEN=False):
    pe = pefile.PE(pe_filename)
    pe_file_data = pe.dump_info()
    pe.dump_dict()  # Make sure that it does not fail
    pe_file_data = pe_file_data.replace("\n\r", "\n")

    control_data_filename = f"{pe_filename}.dmp"

    if REGEN:
        with open(control_data_filename, "wb") as control_data_f:
            control_data_f.write(pe_file_data.encode("utf-8", "backslashreplace"))
        return

    with open(control_data_filename, "rb") as control_data_f:
        control_data = control_data_f.read()

    pe_file_data_hash = sha256(
        pe_file_data.encode("utf-8", "backslashreplace")
    ).hexdigest()
    control_data_hash = sha256(control_data).hexdigest()

    diff_lines_added_count = 0
    diff_lines_removed_count = 0
    lines_to_ignore = 0

    if control_data_hash != pe_file_data_hash:
        print("\nHash differs for [%s]" % os.path.basename(pe_filename))

        control_file_lines = [
            l for l in control_data.decode("utf-8").splitlines()
        ]
        pefile_lines = pe_file_data.splitlines()

        diff = difflib.ndiff(control_file_lines, pefile_lines)

        # check the diff
        for line in diff:
            # Count all changed lines
            if line.startswith("+ "):
                diff_lines_added_count += 1
                # Windows returns slightly different date strings,
                # ignore those.
                if "TimeDateStamp" in line:
                    lines_to_ignore += 1
            if line.startswith("- "):
                diff_lines_removed_count += 1
                # Same as before, the condition is here, in both
                # places because we want to count only the lines in
                # which TimeDateStamp appears that are different, the
                # identical ones are good.
                if "TimeDateStamp" in line:
                    lines_to_ignore += 1

        if (
            diff_lines_removed_count == diff_lines_added_count
            and lines_to_ignore
            == diff_lines_removed_count + diff_lines_added_count
        ):
            print("Differences are in TimeDateStamp formatting, " "ignoring...")

        else:
            assert pe_file_data == control_data.decode("utf-8")


class Test_pefile(unittest.TestCase):

    maxDiff = None

    def test_get_rich_header_hash(self):
        """Verify the RICH_HEADER hashes."""

        control_file = os.path.join(REGRESSION_TESTS_DIR, "kernel32.dll")
        pe = pefile.PE(control_file)

        self.assertEqual(pe.get_rich_header_hash(), "53281e71643c43d225011202b32645d1")
        self.assertEqual(
            pe.get_rich_header_hash("md5"), "53281e71643c43d225011202b32645d1"
        )
        self.assertEqual(
            pe.get_rich_header_hash(algorithm="sha1"),
            "eb7981fdc928971ba400eea3db63ff9e5ec216b1",
        )
        self.assertEqual(
            pe.get_rich_header_hash(algorithm="sha256"),
            "5098ea0fb22f6a21b2806b3cc37d626c2e27593835e44967894636caad49e2d5",
        )
        self.assertEqual(
            pe.get_rich_header_hash(algorithm="sha512"),
            "86044cd48106affa55f4ecf7e1a3c29ecb69fd147085987a2ca1b44aabb8e704"
            "0059570db34b87f56a8359c1847fd3dd406fcf1d0a53fd1981fe519f1b1ede80",
        )
        self.assertRaises(Exception, pe.get_rich_header_hash, algorithm="badalgo")

    def test_selective_loading_integrity(self):
        """Verify integrity of loading the separate elements of the file as
        opposed to do a single pass.
        """

        control_file = os.path.join(REGRESSION_TESTS_DIR, "MSVBVM60.DLL")
        pe = pefile.PE(control_file, fast_load=True)
        # Load the 16 directories.
        pe.parse_data_directories(directories=list(range(0x10)))

        # Do it all at once.
        pe_full = pefile.PE(control_file, fast_load=False)

        # Verify both methods obtained the same results.
        self.assertEqual(pe_full.dump_info(), pe.dump_info())

        pe.close()
        pe_full.close()

    def test_imphash(self):
        """Test imphash values."""

        self.assertEqual(
            pefile.PE(os.path.join(REGRESSION_TESTS_DIR, "mfc40.dll")).get_imphash(),
            "b0f969ff16372d95ef57f05aa8f69409",
        )

        self.assertEqual(
            pefile.PE(os.path.join(REGRESSION_TESTS_DIR, "kernel32.dll")).get_imphash(),
            "437d147ea3f4a34fff9ac2110441696a",
        )

        self.assertEqual(
            pefile.PE(
                os.path.join(
                    REGRESSION_TESTS_DIR,
                    "66c74e4c9dbd1d33b22f63cd0318b72dea88f9dbb4d36a3383d3da20b037d42e",
                )
            ).get_imphash(),
            "a781de574e0567285ee1233bf6a57cc0",
        )

        self.assertEqual(
            pefile.PE(os.path.join(REGRESSION_TESTS_DIR, "cmd.exe")).get_imphash(),
            "d0058544e4588b1b2290b7f4d830eb0a",
        )

    def test_write_header_fields(self):
        """Verify correct field data modification."""

        # Test version information writing
        control_file = os.path.join(REGRESSION_TESTS_DIR, "MSVBVM60.DLL")
        pe = pefile.PE(control_file, fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]]
        )

        original_data = pe.write()

        str1 = b"string1"
        str2 = b"str2"
        str3 = b"string3"

        pe.FileInfo[0][0].StringTable[0].entries[b"FileDescription"] = str1
        pe.FileInfo[0][0].StringTable[0].entries[b"FileVersion"] = str2
        pe.FileInfo[0][0].StringTable[0].entries[b"InternalName"] = str3

        new_data = pe.write()

        diff, differences = 0, list()
        for idx in range(len(original_data)):
            if original_data[idx] != new_data[idx]:

                diff += 1
                # Skip the zeroes that pefile automatically adds to pad a new,
                # shorter string, into the space occupied by a longer one.
                if new_data[idx] != 0:
                    differences.append(chr(new_data[idx]))

        # Verify all modifications in the file were the ones we just made
        #
        self.assertEqual(
            "".join(differences).encode("utf-8", "backslashreplace"), str1 + str2 + str3
        )

        pe.close()

    def test_nt_headers_exception(self):
        """pefile should fail parsing invalid data (missing NT headers)"""

        # Take a known good file.
        control_file = os.path.join(REGRESSION_TESTS_DIR, "MSVBVM60.DLL")
        pe = pefile.PE(control_file, fast_load=True)

        # Truncate it at the PE header and add invalid data.
        pe_header_offest = pe.DOS_HEADER.e_lfanew
        corrupted_data = pe.__data__[:pe_header_offest] + b"\0" * (1024 * 10)

        self.assertRaises(pefile.PEFormatError, pefile.PE, data=corrupted_data)

    def test_dos_header_exception_large_data(self):
        """pefile should fail parsing 10KiB of invalid data
        (missing DOS header).
        """

        # Generate 10KiB of zeroes
        data = b"\0" * (1024 * 10)

        # Attempt to parse data and verify PE header, a PEFormatError exception
        # is thrown.
        self.assertRaises(pefile.PEFormatError, pefile.PE, data=data)

    def test_dos_header_exception_small_data(self):
        """pefile should fail parsing 64 bytes of invalid data
        (missing DOS header).
        """

        # Generate 64 bytes of zeroes
        data = b"\0" * (64)

        # Attempt to parse data and verify PE header a PEFormatError exception
        # is thrown.
        self.assertRaises(pefile.PEFormatError, pefile.PE, data=data)

    def test_empty_file_exception(self):
        """pefile should fail parsing empty files."""

        # Take a known good file
        control_file = os.path.join(REGRESSION_TESTS_DIR, "empty_file")
        self.assertRaises(pefile.PEFormatError, pefile.PE, control_file)

    def test_relocated_memory_mapped_image(self):
        """Test different rebasing methods produce the same image"""

        # Take a known good file
        control_file = os.path.join(REGRESSION_TESTS_DIR, "MSVBVM60.DLL")
        pe = pefile.PE(control_file)

        def count_differences(data1, data2):
            diff = 0
            for idx in range(len(data1)):
                if data1[idx] != data2[idx]:
                    diff += 1
            return diff

        original_image_1 = pe.get_memory_mapped_image()
        rebased_image_1 = pe.get_memory_mapped_image(ImageBase=0x1000000)

        differences_1 = count_differences(original_image_1, rebased_image_1)
        self.assertEqual(differences_1, 61136)

        pe = pefile.PE(control_file)

        original_image_2 = pe.get_memory_mapped_image()
        pe.relocate_image(0x1000000)
        rebased_image_2 = pe.get_memory_mapped_image()

        differences_2 = count_differences(original_image_2, rebased_image_2)
        self.assertEqual(differences_2, 61136)

        # Ensure the original image stayed the same
        self.assertEqual(original_image_1, original_image_2)

        # This file used to crash pefile when attempting to relocate it:
        # https://github.com/erocarrera/pefile/issues/314
        control_file = os.path.join(
            REGRESSION_TESTS_DIR, "pefile-314/crash-8499a0bb33aeba8f59a172584abc7ca0ab82a78c"
        )
        pe = pefile.PE(control_file)

    def test_checksum(self):
        """Verify correct calculation of checksum"""

        # Take a known good file.
        control_file = os.path.join(REGRESSION_TESTS_DIR, "MSVBVM60.DLL")
        pe = pefile.PE(control_file)

        # verify_checksum() generates a checksum from the image's data and
        # compares it against the checksum field in the optional header.
        self.assertEqual(pe.verify_checksum(), True)
