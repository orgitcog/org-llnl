# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""
Sonar Directory Source Connector Configuration Class
"""

import os
import json
from enum import Enum

from sonar_driver.kafka_connect.connector_config import ConnectorConfig


class FileFormat(Enum):
    JSON    = 'json'
    CSV     = 'csv'

    def __str__(self):
        return str(self.value)


class SonarDirectorySourceConfig(ConnectorConfig):

    CONNECTOR_CLASS         = "gov.llnl.sonar.kafka.connect.connectors.DirectorySourceConnector"

    TOPIC_KEY               = "topic"
    BATCH_ROWS_KEY          = "batch.rows"
    BATCH_FILES_KEY         = "batch.files"
    DIRNAME_KEY             = "dirname"
    COMPLETED_DIRNAME_KEY   = "completed.dirname" 
    FORMAT_KEY              = "format" 
    FORMAT_OPTIONS_KEY      = "format.options" 
    AVRO_SCHEMA_KEY         = "avro.schema"
    ZK_HOST_KEY             = "zk.host"
    ZK_PORT_KEY             = "zk.port"

    def __init__(self, 
            topic, 
            dirname, 
            completed_dirname, 
            avro_schema,
            tasks_max=1, 
            file_format=FileFormat.JSON, 
            format_options={}, 
            batch_rows=10000,
            batch_files=10,
            zookeeper_host="localhost",
            zookeeper_port=2181):

        super().__init__(tasks_max)

        dirname_abspath = os.path.abspath(dirname)
        completed_dirname_abspath = os.path.abspath(completed_dirname)

        self.config_dict[self.TOPIC_KEY]                = topic
        self.config_dict[self.BATCH_ROWS_KEY]           = batch_rows
        self.config_dict[self.BATCH_FILES_KEY]          = batch_files
        self.config_dict[self.DIRNAME_KEY]              = dirname_abspath
        self.config_dict[self.COMPLETED_DIRNAME_KEY]    = completed_dirname_abspath
        self.config_dict[self.FORMAT_KEY]               = file_format
        self.config_dict[self.FORMAT_OPTIONS_KEY]       = format_options
        self.config_dict[self.AVRO_SCHEMA_KEY]          = json.dumps(avro_schema.to_json())
        self.config_dict[self.ZK_HOST_KEY]              = zookeeper_host
        self.config_dict[self.ZK_PORT_KEY]              = zookeeper_port

