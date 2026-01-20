// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import gov.llnl.sonar.kafka.connect.offsetmanager.FileOffset;
import gov.llnl.sonar.kafka.connect.parsers.FileStreamParser;
import gov.llnl.sonar.kafka.connect.parsers.FileStreamParserBuilder;
import gov.llnl.sonar.kafka.connect.parsers.ParseException;
import gov.llnl.sonar.kafka.connect.util.VersionUtil;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.connect.source.SourceRecord;
import org.apache.kafka.connect.source.SourceTask;
import org.json.JSONObject;

import java.io.EOFException;
import java.io.File;
import java.net.InetAddress;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Slf4j
public class FileSourceTask extends SourceTask {

    private String taskID;
    private static final String PARTITION_FIELD = "fileName";
    private static final String OFFSET_FIELD = "line";

    private FileSourceConfig config;
    private FileStreamParser streamParser;

    @Override
    public String version() {
        return VersionUtil.getVersion();
    }

    @Override
    public void start(Map<String, String> map) {

        config = new FileSourceConfig(map);

        try {
            // Get local task id
            this.taskID = InetAddress.getLocalHost().getHostName() + "(" + Thread.currentThread().getId() + ")";

            // Parse avro schema
            org.apache.avro.Schema avroSchema;
            if (!config.getAvroSchema().isEmpty()) {
                avroSchema = new org.apache.avro.Schema.Parser().parse(config.getAvroSchema());
            } else {
                avroSchema = new org.apache.avro.Schema.Parser().parse(new File(config.getAvroSchemaFilename()));
            }

            // Build FileStreamParser
            FileStreamParserBuilder fileStreamParserBuilder = new FileStreamParserBuilder();
            fileStreamParserBuilder.setAvroSchema(avroSchema);
            fileStreamParserBuilder.setFormat(config.getFormat());
            fileStreamParserBuilder.setFormatOptions(new JSONObject(config.getFormatOptions()));
            fileStreamParserBuilder.setEofSentinel(config.getEofSentinel());
            fileStreamParserBuilder.setPartitionField(PARTITION_FIELD);
            fileStreamParserBuilder.setOffsetField(OFFSET_FIELD);
            this.streamParser = fileStreamParserBuilder.build(
                    Paths.get(config.getFilename()),
                    new FileOffset(0L, 0L, true, false));

            log.info("Task {}: Added ingestion file {}", taskID, config.getFilename());

        } catch (EOFException e){
            log.info("Task {}: EOF reached for file {}", taskID, config.getFilename());
        } catch (Exception ex) {
            log.error("Task {}: ", taskID, ex);
        }
    }

    @Override
    public List<SourceRecord> poll() throws InterruptedException {

        List<SourceRecord> records = new ArrayList<>();

        try {
            for (int i = 0; i < config.getBatchSize(); i++) {
                try {
                    records.add(streamParser.readNextRecord(config.getTopic()));
                } catch (EOFException e) {
                    log.info("Task {}: {}", taskID, e.getMessage());
                    break;
                } catch (ParseException e) {
                    log.warn("Task {}: {}", taskID, e.getMessage());
                }
            }
        } catch (Exception e) {
            log.error("Task {}: ", taskID, e);
            synchronized (this) {
                this.wait(1000);
            }
        }

        // If empty, return null
        if (records.isEmpty()) {
            records = null;
            synchronized (this) {
                this.wait(1000);
            }
        } else {
            log.info("Task {}: Read {} records from file {}", taskID, records.size(), streamParser.getFileName());
        }

        return records;
    }

    @Override
    public void stop() {
        log.info("Task {}: Stopping", taskID);
        try {
            streamParser.close();
            synchronized (this) {
                this.notify();
            }
        } catch (Exception ex) {
            log.error("Task {}: {}", taskID, ex);
        }
    }
}