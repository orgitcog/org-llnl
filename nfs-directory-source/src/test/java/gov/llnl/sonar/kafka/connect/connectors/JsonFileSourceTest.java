// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FilenameUtils;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static gov.llnl.sonar.kafka.connect.connectors.ConnectTestData.idstrAvroData;
import static gov.llnl.sonar.kafka.connect.connectors.ConnectTestData.idstrAvroSchemaEscapedString;

@Log4j2
public class JsonFileSourceTest extends ConnectTest {

    Map<String, String> configProperties = new HashMap<>();

    private Path jsonTestFile;
    private Path outputDir;
    private String jsonTestSourceConnector;
    private String jsonTestSourceTopic;

    @BeforeAll
    public void createTestContents() {

        try {
            log.info("Creating test JSON file");
            jsonTestFile = Files.createTempFile("json-test-file-source-", ".json");

            log.info("Writing JSON entries to file source");
            BufferedWriter bw = new BufferedWriter(new FileWriter(jsonTestFile.toFile()));
            bw.write("{\"id\": 1, \"str\": \"one\"}\n");
            bw.write("{\"id\": 2, \"str\": \"two\"}\n");
            bw.write("{\"id\": 3, \"str\": \"three\"}\n");
            bw.write("{\"id\": 4, \"str\": \"four\"}\n");
            bw.write("{\"id\": 5, \"str\": \"five\"}\n");
            bw.write("{\"id\": 6, \"str\": \"six\"}\n");
            bw.write("{\"id\": 7, \"str\": \"seven\"}\n");
            bw.write("{\"id\": 8, \"str\": \"eight\"}\n");
            bw.flush();

            outputDir = Files.createTempDirectory("outputDir");
        } catch (IOException ex) {
            log.error(ex);
        }

        String jsonTestFilename = jsonTestFile.toString();
        String jsonTestFileBasename = FilenameUtils.getBaseName(jsonTestFilename);
        jsonTestSourceConnector = jsonTestFileBasename;
        jsonTestSourceTopic = jsonTestFileBasename + "-topic";

        configProperties.put(FileSourceConfig.FILENAME, jsonTestFilename);
        configProperties.put(FileSourceConfig.FORMAT, "json");
        configProperties.put(FileSourceConfig.FORMAT_OPTIONS, "{}");
        configProperties.put(FileSourceConfig.COMPLETED_DIRNAME, outputDir.toAbsolutePath().toString());
        configProperties.put(FileSourceConfig.TOPIC, jsonTestSourceTopic);
        configProperties.put(FileSourceConfig.AVRO_SCHEMA, idstrAvroSchemaEscapedString);

    }

    @Test
    public void testFileSourceJson() throws IOException {

        log.info("Creating connector " + jsonTestSourceConnector);
        log.info(confluent.createConnector(jsonTestSourceConnector, FileSourceConnector.class, configProperties));

        validateTopicContents(jsonTestSourceTopic, idstrAvroData);
    }

    @AfterAll
    public void deleteTestContents() {
        confluent.deleteConnector(jsonTestSourceConnector);
        confluent.deleteTopic(jsonTestSourceTopic);
        jsonTestFile.toFile().delete();
    }

}