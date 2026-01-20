// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import org.apache.kafka.common.config.AbstractConfig;
import org.apache.kafka.common.config.ConfigDef;
import org.apache.kafka.common.config.ConfigDef.Importance;
import org.apache.kafka.common.config.ConfigDef.Type;

import java.util.Map;


public class FileSourceConfig extends AbstractConfig {

    public FileSourceConfig(ConfigDef config, Map<String, String> parsedConfig) {
        super(config, parsedConfig);
    }
    public FileSourceConfig(Map<String, String> parsedConfig) {
        this(conf(), parsedConfig);
    }

    public static final String FILENAME = "fileName";
    public static final String FILENAME_DOC = "The file to readNextRecord from.";
    public static final String COMPLETED_DIRNAME = "completed.dirname";
    public static final String COMPLETED_DIRNAME_DOC = "The directory to move ingested files into";
    public static final String TOPIC = "topic";
    public static final String TOPIC_DOC = "The name of the topic to stream to.";
    public static final String FORMAT = "format";
    public static final String FORMAT_DOC = "Format of the file [ csv | json ]";
    public static final String FORMAT_OPTIONS = "format.options";
    public static final String FORMAT_OPTIONS_DOC = "Comma-separated list of formatting options as option:value.\n" +
            "Available options:\n" +
            "   csv: header:[true|false],delim:<char>,quote=<char>\n" +
            "   json: orient:[records]" ;
    public static final String AVRO_SCHEMA = "avro.schema";
    public static final String AVRO_SCHEMA_DOC = "Avro schema string, e.g., " +
            "{\n" +
            "  \"type\": \"record\",\n" +
            "  \"name\": \"idstr\",\n" +
            "  \"fields\" : [\n" +
            "    {\"name\": \"id\", \"type\": \"int\"},\n" +
            "    {\"name\": \"str\", \"type\": \"string\"},\n" +
            "  ]\n" +
            "}";
    public static final String AVRO_SCHEMA_FILENAME = "avro.schema.fileName";
    public static final String AVRO_SCHEMA_FILENAME_DOC = "Avro schema fileName.";
    public static final String BATCH_SIZE = "batch.size";
    public static final String BATCH_SIZE_DOC = "Number of lines to readNextRecord/ingest at a time from the file.";
    public static final String EOF_SENTINEL = "eof.sentinel";
    public static final String EOF_SENTINEL_DOC = "String indicating the end of a file." +
            "If defined, files will not be purged until this is reached";

    public static ConfigDef conf() {
        return new ConfigDef()
                .define(FILENAME, Type.STRING, Importance.HIGH, FILENAME_DOC)
                .define(COMPLETED_DIRNAME, Type.STRING, Importance.HIGH, COMPLETED_DIRNAME_DOC)
                .define(TOPIC, Type.STRING, Importance.HIGH, TOPIC_DOC)
                .define(FORMAT, Type.STRING, Importance.HIGH, FORMAT_DOC)
                .define(FORMAT_OPTIONS, Type.STRING, "{}", Importance.LOW, FORMAT_OPTIONS_DOC)
                .define(AVRO_SCHEMA, Type.STRING, "", Importance.HIGH, AVRO_SCHEMA_DOC)
                .define(AVRO_SCHEMA_FILENAME, Type.STRING, "", Importance.HIGH, AVRO_SCHEMA_FILENAME_DOC)
                .define(BATCH_SIZE, Type.LONG, 1000L, Importance.HIGH, BATCH_SIZE_DOC)
                .define(EOF_SENTINEL, Type.STRING, null, Importance.HIGH, EOF_SENTINEL_DOC)
                ;
    }

    public String getFilename() { return this.getString(FILENAME); }
    public String getCompletedDirname() { return this.getString(COMPLETED_DIRNAME); }
    public String getFormat() { return this.getString(FORMAT); }
    public String getFormatOptions() { return this.getString(FORMAT_OPTIONS); }
    public String getTopic() { return this.getString(TOPIC); }
    public String getAvroSchema() { return this.getString(AVRO_SCHEMA); }
    public String getAvroSchemaFilename() { return this.getString(AVRO_SCHEMA_FILENAME); }
    public Long getBatchSize() { return this.getLong(BATCH_SIZE); }
    public String getEofSentinel() { return this.getString(EOF_SENTINEL); }
}

