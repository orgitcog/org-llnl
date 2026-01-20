// Copyright 2018 Lawrence Livermore National Security, LLC and other
// nfs-directory-source Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: (Apache-2.0 OR MIT)

package gov.llnl.sonar.kafka.connect.connectors;

import gov.llnl.sonar.kafka.connect.util.VersionUtil;
import org.apache.kafka.common.config.Config;
import org.apache.kafka.common.config.ConfigDef;
import org.apache.kafka.common.config.ConfigValue;
import org.apache.kafka.connect.connector.Task;
import org.apache.kafka.connect.source.SourceConnector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class FileSourceConnector extends SourceConnector {

    private FileSourceConfig config;

    @Override
    public String version() {
        return VersionUtil.getVersion();
    }

    @Override
    public void start(Map<String, String> props) {
        config = new FileSourceConfig(props);
    }

    @Override
    public Class<? extends Task> taskClass() {
        return FileSourceTask.class;
    }

    @Override
    public List<Map<String, String>> taskConfigs(int maxTasks) {
        return new ArrayList<>(Collections.nCopies(maxTasks, config.originalsStrings()));
    }

    @Override
    public void stop() { }

    @Override
    public ConfigDef config() {
        return FileSourceConfig.conf();
    }

    @Override
    public Config validate(Map<String, String> connectorConfigs) {
        Config c = super.validate(connectorConfigs);

        List<ConfigValue> configValues = c.configValues();
        if (connectorConfigs.containsKey("avro.schema") == connectorConfigs.containsKey("avro.schema.fileName")) {
            for (ConfigValue cv : configValues) {
                if (cv.name().equals("avro.schema")) {
                    cv.addErrorMessage("Connector requires either avro.schema or avro.schema.fileName (and not both)!");
                }
            }
        }

        return new Config(configValues);
    }
}
