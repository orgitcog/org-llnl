// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.parsers;

import java.text.MessageFormat;

public class ParseException extends Exception {
    ParseException(FileStreamParser fileStreamParser, String message) {
        super(MessageFormat.format(
                "Parse exception at file {0}: offset {1}\n{2}",
                fileStreamParser.fileName,
                fileStreamParser.offset.toString(),
                message));
    }
}
