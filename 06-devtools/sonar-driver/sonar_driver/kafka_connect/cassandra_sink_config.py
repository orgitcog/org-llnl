# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""
Cassandra Sink Connector Configuration Class
"""

from sonar_driver.kafka_connect.connector_config import ConnectorConfig


class CassandraSinkConfig(ConnectorConfig):

    CONNECTOR_CLASS         = "com.datamountaineer.streamreactor.connect.cassandra.sink.CassandraSinkConnector"
    
    TOPICS_KEY              = "topics"
    CASSANDRA_KEYSPACE_KEY  = "connect.cassandra.key.space"
    CASSANDRA_HOST_KEY      = "connect.cassandra.contact.points"
    CASSANDRA_PORT_KEY      = "connect.cassandra.port"
    CASSANDRA_USERNAME_KEY  = "connect.cassandra.username"
    CASSANDRA_PASSFILE_KEY  = "connect.cassandra.password.file"
    KCQL_KEY                = "connect.cassandra.kcql"
    DELETE_SUNK_KEY         = "delete.sunk.kafka.records"

    def __init__(self, 
            topics, 
            keyspace,
            table,
            cassandra_username,
            cassandra_password_file,
            cassandra_hosts=['localhost'],
            cassandra_port=9042,
            kcql=None, # will be inferred
            tasks_max=1,
            delete_sunk_kafka_records=True):

        super().__init__(tasks_max)

        self.config_dict[self.TOPICS_KEY]   		= topics
        self.config_dict[self.CASSANDRA_KEYSPACE_KEY]   = keyspace
        self.config_dict[self.CASSANDRA_HOST_KEY]       = ','.join(cassandra_hosts)
        self.config_dict[self.CASSANDRA_PORT_KEY]       = cassandra_port
        self.config_dict[self.CASSANDRA_USERNAME_KEY]   = cassandra_username
        self.config_dict[self.CASSANDRA_PASSFILE_KEY]   = cassandra_password_file
        self.config_dict[self.DELETE_SUNK_KEY]          = delete_sunk_kafka_records

        if kcql is None:
            self.config_dict[self.KCQL_KEY] = "INSERT INTO " + table + " SELECT * FROM " + topics
        else:
            self.config_dict[self.KCQL_KEY] = kcql

