// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import lombok.extern.log4j.Log4j2;
import org.apache.avro.Schema;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecordBuilder;

import java.util.*;

@Log4j2
public class ConnectTestData {

    static String idstrAvroSchemaString = "{" +
            "  \"type\": \"record\"," +
            "  \"name\": \"idstr\"," +
            "  \"connect.name\": \"idstr\"," +
            "  \"fields\": [" +
            "    {" +
            "      \"name\": \"id\"," +
            "      \"type\": \"int\"" +
            "    }," +
            "    {" +
            "      \"name\": \"str\"," +
            "      \"type\": \"string\"" +
            "    }" +
            "  ]" +
            "}";

    static String idstrAvroSchemaEscapedString = idstrAvroSchemaString.replaceAll("\"", "\\\"");

    static Schema idstrAvroSchema = new Schema.Parser().parse(idstrAvroSchemaString);

    static Set<GenericData.Record> idstrAvroData = new HashSet<>(Arrays.asList(
            new GenericRecordBuilder(idstrAvroSchema).set("id", 1).set("str", "one").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 2).set("str", "two").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 3).set("str", "three").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 4).set("str", "four").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 5).set("str", "five").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 6).set("str", "six").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 7).set("str", "seven").build(),
            new GenericRecordBuilder(idstrAvroSchema).set("id", 8).set("str", "eight").build()
            ));

}
