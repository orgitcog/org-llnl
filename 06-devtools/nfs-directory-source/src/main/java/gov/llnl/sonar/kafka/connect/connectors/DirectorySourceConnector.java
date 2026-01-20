// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import gov.llnl.sonar.kafka.connect.offsetmanager.FileOffsetManager;
import gov.llnl.sonar.kafka.connect.util.VersionUtil;
import lombok.extern.slf4j.Slf4j;
import org.apache.kafka.common.config.Config;
import org.apache.kafka.common.config.ConfigDef;
import org.apache.kafka.common.config.ConfigValue;
import org.apache.kafka.connect.connector.Task;
import org.apache.kafka.connect.source.SourceConnector;

import java.net.InetAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


@Slf4j
public class DirectorySourceConnector extends SourceConnector {

    private String taskID;
    private DirectorySourceConfig config;
    private static long BATCH_SIZE_MAX = 500000L;

    @Override
    public String version() {
        return VersionUtil.getVersion();
    }

    @Override
    public void start(Map<String, String> props) {

        // Make dirname absolute
        Path absolutePath = Paths.get(props.get(DirectorySourceConfig.DIRNAME)).toAbsolutePath();
        String absoluteDirname = absolutePath.toString();
        props.put(DirectorySourceConfig.DIRNAME, absoluteDirname);

        // if (Long.valueOf(props.get(DirectorySourceConfig.BATCH_ROWS)) > BATCH_SIZE_MAX) {
        //     log.warn(String.format("Specified \"%s\" exceeds maximum %d, setting to maximum", DirectorySourceConfig.BATCH_SIZE, BATCH_SIZE_MAX));
        //     props.put(DirectorySourceConfig.BATCH_SIZE, String.valueOf(BATCH_SIZE_MAX));
        // }

        config = new DirectorySourceConfig(props);

        try {
            // Get task ID
            this.taskID = InetAddress.getLocalHost().getHostName() + "(" + Thread.currentThread().getId() + ")";
            log.info("Connector task {}: Start", taskID);
        } catch (Exception e) {
            log.error("Exception:", e);
        }
    }

    @Override
    public Class<? extends Task> taskClass() {
        return DirectorySourceTask.class;
    }

    @Override
    public List<Map<String, String>> taskConfigs(int maxTasks) {
        log.info("Connector task {}: Creating {} directory source tasks", taskID, maxTasks);

        Path absolutePath = Paths.get(config.getDirname()).toAbsolutePath();

        Map<String, String> configStrings = config.originalsStrings();
        configStrings.put("zk.fileOffsetPath", absolutePath.toString());

        return new ArrayList<>(Collections.nCopies(maxTasks, configStrings));
    }

    @Override
    public void stop() {
        log.info("Connector task {}: Stop", taskID);
    }

    @Override
    public ConfigDef config() {
        return DirectorySourceConfig.conf();
    }

    @Override
    public Config validate(Map<String, String> connectorConfigs) {
        Config c = super.validate(connectorConfigs);

        List<ConfigValue> configValues = c.configValues();

        // Must have valid dir
        String dirname = connectorConfigs.get(DirectorySourceConfig.DIRNAME);
        Path dirpath = Paths.get(dirname);

        if (!(  Files.exists(dirpath) &&
                Files.isDirectory(dirpath) &&
                Files.isReadable(dirpath) &&
                Files.isExecutable(dirpath))) {
            for (ConfigValue cv : configValues) {
                if (cv.name().equals(DirectorySourceConfig.DIRNAME)) {
                    cv.addErrorMessage("Specified \"" + DirectorySourceConfig.DIRNAME +  "\" must: exist, be a directory, be readable and executable");
                }
            }
        }

        // Must have valid competed dir
        // TODO: allow for delete-on-ingest without a specified completed dir
        String completeddirname = connectorConfigs.get(DirectorySourceConfig.COMPLETED_DIRNAME);
        if (completeddirname != null) {
            Path completeddirpath = Paths.get(completeddirname);

            if (!(Files.exists(completeddirpath) &&
                    Files.isDirectory(completeddirpath) &&
                    Files.isWritable(completeddirpath) &&
                    Files.isExecutable(completeddirpath))) {
                for (ConfigValue cv : configValues) {
                    if (cv.name().equals(DirectorySourceConfig.COMPLETED_DIRNAME)) {
                        cv.addErrorMessage("Specified \"" + DirectorySourceConfig.COMPLETED_DIRNAME + "\" must: exist, be a directory, be writable and executable");
                    }
                }
            }
        }

        // Must have avro.schema or avro.schema.fileName
        if (connectorConfigs.containsKey(DirectorySourceConfig.AVRO_SCHEMA) == connectorConfigs.containsKey(DirectorySourceConfig.AVRO_SCHEMA_FILENAME)) {
            for (ConfigValue cv : configValues) {
                if (cv.name().equals(DirectorySourceConfig.AVRO_SCHEMA)) {
                    cv.addErrorMessage("Connector requires either \"" + DirectorySourceConfig.AVRO_SCHEMA + "\" or \"" + DirectorySourceConfig.AVRO_SCHEMA_FILENAME + "\" (and not both)!");
                }
            }
        }

        // This validation is run once per connector installation, so here we also initialize the file offset manager
        // with no contents (reset=true).
        // The DirectorySourceTask instances will then use the initialized file offset Zookeeper nodes
        // without having to create them.
        try {
            log.info("{}: creating file offset manager", this.getClass());

            // Get configs
            Path absolutePath = Paths.get(connectorConfigs.get(DirectorySourceConfig.DIRNAME)).toAbsolutePath();
            String absoluteDirname = absolutePath.toString();
            String zooKeeperHost = connectorConfigs.get(DirectorySourceConfig.ZKHOST);
            String zooKeeperPort = connectorConfigs.get(DirectorySourceConfig.ZKPORT);

            // Create and reset file getByteOffset manager
            FileOffsetManager fileOffsetManager = new FileOffsetManager(zooKeeperHost, zooKeeperPort, absoluteDirname, true);
            fileOffsetManager.close();
        } catch (Exception e) {
            log.error("Exception:", e);
        }


        return new Config(configValues);
    }
}

