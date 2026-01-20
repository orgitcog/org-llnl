// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

import static gov.llnl.sonar.kafka.connect.connectors.ConnectTestData.idstrAvroData;
import static gov.llnl.sonar.kafka.connect.connectors.ConnectTestData.idstrAvroSchemaEscapedString;

@Log4j2
public class DirectorySourceTest extends ConnectTest {

    Map<String, String> configProperties = new HashMap<>();

    private Path testDirectory;
    private Path outputDirectory;
    private String testDirectorySourceConnector;
    private String testDirectorySourceTopic;

    @BeforeAll
    public void createTestContents() {

        try {

            log.info("Creating test directory");
            testDirectory = Files.createTempDirectory("test-directory-source-");

            log.info("Creating files in test directory");
            File testFile = File.createTempFile("json-test-file-", ".json", testDirectory.toFile());
            BufferedWriter bw = new BufferedWriter(new FileWriter(testFile));
            bw.write("{\"id\": 1, \"str\": \"one\"}\n");
            bw.write("{\"id\": 2, \"str\": \"two\"}\n");
            bw.flush();

            File testFile2 = File.createTempFile("json-test-file-", ".json", testDirectory.toFile());
            BufferedWriter bw2 = new BufferedWriter(new FileWriter(testFile2));
            bw2.write("{\"id\": 3, \"str\": \"three\"}\n");
            bw2.write("{\"id\": 4, \"str\": \"four\"}\n");
            bw2.flush();

            File testFile3 = File.createTempFile("json-test-file-", ".json", testDirectory.toFile());
            BufferedWriter bw3 = new BufferedWriter(new FileWriter(testFile3));
            bw3.write("{\"id\": 5, \"str\": \"five\"}\n");
            bw3.write("{\"id\": 6, \"str\": \"six\"}\n");
            bw3.flush();

            File testFile4 = File.createTempFile("json-test-file-", ".json", testDirectory.toFile());
            BufferedWriter bw4 = new BufferedWriter(new FileWriter(testFile4));
            bw4.write("{\"id\": 7, \"str\": \"seven\"}\n");
            bw4.write("{\"id\": 8, \"str\": \"eight\"}\n");
            bw4.flush();

            outputDirectory = Files.createTempDirectory("outputDir");

        } catch (IOException ex) {
            log.error(ex);
        }

        String testDirname = testDirectory.toString();
        String testDirBasename = FilenameUtils.getBaseName(testDirname);
        testDirectorySourceConnector = testDirBasename;
        testDirectorySourceTopic = testDirBasename + "-topic";

        configProperties.put("tasks.max", "3");
        configProperties.put(DirectorySourceConfig.BATCH_FILES, "2");
        configProperties.put(DirectorySourceConfig.BATCH_ROWS, "2");
        configProperties.put(DirectorySourceConfig.DIRNAME, testDirname);
        configProperties.put(DirectorySourceConfig.FORMAT, "json");
        configProperties.put(DirectorySourceConfig.FORMAT_OPTIONS, "{}");
        configProperties.put(DirectorySourceConfig.COMPLETED_DIRNAME, outputDirectory.toAbsolutePath().toString());
        configProperties.put(DirectorySourceConfig.TOPIC, testDirectorySourceTopic);
        configProperties.put(DirectorySourceConfig.AVRO_SCHEMA, idstrAvroSchemaEscapedString);
        configProperties.put(DirectorySourceConfig.ZKHOST, "localhost");
        configProperties.put(DirectorySourceConfig.ZKPORT, "2181");

    }

    @Test
    public void testFileSourceJson() throws IOException {

        log.info("Creating connector " + testDirectorySourceConnector);
        log.info(confluent.createConnector(testDirectorySourceConnector, DirectorySourceConnector.class, configProperties));

        validateTopicContents(testDirectorySourceTopic, idstrAvroData);
    }

    @AfterAll
    public void deleteTestContents() {
        confluent.deleteConnector(testDirectorySourceConnector);
        confluent.deleteTopic(testDirectorySourceTopic);
        try {
            FileUtils.deleteDirectory(testDirectory.toFile());
        } catch (IOException ex) {
            log.error(ex);
        }
    }

}