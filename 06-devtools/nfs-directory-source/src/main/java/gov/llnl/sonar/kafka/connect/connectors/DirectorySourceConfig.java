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

public class DirectorySourceConfig extends AbstractConfig {

    public DirectorySourceConfig(ConfigDef config, Map<String, String> parsedConfig) {
        super(config, parsedConfig);
    }
    public DirectorySourceConfig(Map<String, String> parsedConfig) {
        this(conf(), parsedConfig);
    }


    public static final String DIRNAME = "dirname";
    public static final String DIRNAME_DOC = "The directory to readNextRecord from.";
    public static final String COMPLETED_DIRNAME = "completed.dirname";
    public static final String COMPLETED_DIRNAME_DOC = "The directory to move ingested files into.";
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
    public static final String BATCH_ROWS = "batch.rows";
    public static final String BATCH_ROWS_DOC = "Number of rows to readNextRecord/ingest at a time from each file.";
    public static final String BATCH_FILES = "batch.files";
    public static final String BATCH_FILES_DOC = "Number of files to readNextRecord/ingest at a time.";
    public static final String ZKHOST = "zk.host";
    public static final String ZKHOST_DOC = "ZooKeeper hostname";
    public static final String ZKPORT = "zk.port";
    public static final String ZKPORT_DOC = "ZooKeeper port";
    public static final String EOF_SENTINEL = "eof.sentinel";
    public static final String EOF_SENTINEL_DOC = "String indicating the end of a file. " +
                                                  "If defined, files will not be purged until this is reached.";
    public static final String DELETE_INGESTED = "delete.ingested";
    public static final String DELETE_INGESTED_DOC = "Whether to delete files after ingesting them.";

    public static ConfigDef conf() {
        return new ConfigDef()
                .define(ZKHOST, Type.STRING, "localhost", Importance.HIGH, ZKHOST_DOC)
                .define(ZKPORT, Type.STRING, "2181", Importance.HIGH, ZKPORT_DOC)
                .define(DIRNAME, Type.STRING, Importance.HIGH, DIRNAME_DOC)
                .define(COMPLETED_DIRNAME, Type.STRING, null, Importance.HIGH, COMPLETED_DIRNAME_DOC)
                .define(TOPIC, Type.STRING, Importance.HIGH, TOPIC_DOC)
                .define(FORMAT, Type.STRING, Importance.HIGH, FORMAT_DOC)
                .define(FORMAT_OPTIONS, Type.STRING, "{}", Importance.LOW, FORMAT_OPTIONS_DOC)
                .define(AVRO_SCHEMA, Type.STRING, "", Importance.HIGH, AVRO_SCHEMA_DOC)
                .define(AVRO_SCHEMA_FILENAME, Type.STRING, "", Importance.HIGH, AVRO_SCHEMA_FILENAME_DOC)
                .define(BATCH_ROWS, Type.LONG, 1000L, Importance.HIGH, BATCH_ROWS_DOC)
                .define(BATCH_FILES, Type.LONG, 10L, Importance.HIGH, BATCH_FILES_DOC)
                .define(EOF_SENTINEL, Type.STRING, null, Importance.HIGH, EOF_SENTINEL_DOC)
                .define(DELETE_INGESTED, Type.BOOLEAN, false, Importance.HIGH, DELETE_INGESTED_DOC)
                ;
    }

    public String getZooKeeperHost() { return this.getString(ZKHOST); }
    public String getZooKeeperPort() { return this.getString(ZKPORT); }
    public String getDirname() { return this.getString(DIRNAME); }
    public String getCompletedDirname() { return this.getString(COMPLETED_DIRNAME); }
    public String getTopic() { return this.getString(TOPIC); }
    public String getFormat() { return this.getString(FORMAT); }
    public String getFormatOptions() { return this.getString(FORMAT_OPTIONS); }
    public String getAvroSchema() { return this.getString(AVRO_SCHEMA); }
    public String getAvroSchemaFilename() { return this.getString(AVRO_SCHEMA_FILENAME); }
    public Long getBatchRows() { return this.getLong(BATCH_ROWS); }
    public Long getBatchFiles() { return this.getLong(BATCH_FILES); }
    public String getEofSentinel() { return this.getString(EOF_SENTINEL); }
    public boolean getDeleteIngested() { return this.getBoolean(DELETE_INGESTED); }
}

