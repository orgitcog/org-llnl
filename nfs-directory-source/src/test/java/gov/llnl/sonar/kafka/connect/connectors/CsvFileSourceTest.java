// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import lombok.extern.log4j.Log4j2;
import org.apache.commons.io.FilenameUtils;
import org.junit.jupiter.api.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static gov.llnl.sonar.kafka.connect.connectors.ConnectTestData.*;

@Log4j2
public class CsvFileSourceTest extends ConnectTest {

    private Path csvTestFile;
    private Path outputDir;
    private String csvTestSourceConnector;
    private String csvTestSourceTopic;

    private Map<String, String> configProperties = new HashMap<>();

    @BeforeAll
    public void createTestContents() {

        try {
            log.info("Creating test CSV file");
            csvTestFile = Files.createTempFile("csv-test-file-source-", ".csv");

            log.info("Writing CSV entries to file source");
            BufferedWriter bw = new BufferedWriter(new FileWriter(csvTestFile.toFile()));
            bw.write("id,str\n"); // header
            bw.write("1,one\n");
            bw.write("2,two\n");
            bw.write("3,three\n");
            bw.write("4,four\n");
            bw.write("5,five\n");
            bw.write("6,six\n");
            bw.write("7,seven\n");
            bw.write("8,eight\n");
            bw.flush();

            outputDir = Files.createTempDirectory("outputDir");
        } catch (IOException ex) {
            log.error(ex);
        }

        String csvTestFilename = csvTestFile.toString();
        String csvTestFileBasename = FilenameUtils.getBaseName(csvTestFilename);
        csvTestSourceConnector = csvTestFileBasename;
        csvTestSourceTopic = csvTestFileBasename + "-topic";

        configProperties.put(FileSourceConfig.FILENAME, csvTestFilename);
        configProperties.put(FileSourceConfig.FORMAT, "csv");
        configProperties.put(FileSourceConfig.FORMAT_OPTIONS, "{ \"withHeader\": true }");
        configProperties.put(FileSourceConfig.TOPIC, csvTestSourceTopic);
        configProperties.put(FileSourceConfig.AVRO_SCHEMA, idstrAvroSchemaEscapedString);
        configProperties.put(FileSourceConfig.COMPLETED_DIRNAME, outputDir.toAbsolutePath().toString());

    }

    @Test
    public void testFileSourceCsv() throws IOException {

        log.info("Creating connector " + csvTestSourceConnector);
        log.info(confluent.createConnector(csvTestSourceConnector, FileSourceConnector.class, configProperties));

        validateTopicContents(csvTestSourceTopic, idstrAvroData);
    }

    @AfterAll
    public void deleteTestContents() {
        confluent.deleteConnector(csvTestSourceConnector);
        confluent.deleteTopic(csvTestSourceTopic);
        csvTestFile.toFile().delete();
    }

}

