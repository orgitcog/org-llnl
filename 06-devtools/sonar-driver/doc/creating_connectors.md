# Creating Connectors

Connectors are JSON objects that can be created using the connector creator scripts.
Created connectors can be installed to a running instance of Kafka Connect using the `install_connectors` script (see [Installing and Uninstalling Connectors](./installing_connectors.md)).

## Connector Creator Scripts

Currently the available connector creator scripts are:

1. `create_cassandra_sink`
2. `create_sonar_directory_source`

## Sources and Sinks

A source sends data to a Kafka topic, and a sink consumes data from a Kafka topic.
In order to connect a source to a sink, you must give them both the same topic name.
Topic names will be uniquely auto-generated if none is provided via the `--topic-name/-n` option.

## Directory Source Example

The following command:

```bash
create_directory_source_connector myingestdir mycompleteddir ./tests/idstr.avsc
```

will produce:

```bash
{
    "config": {
        "avro.schema": "{\"type\": \"record\", \"name\": \"test\", \"fields\": [{\"type\": \"int\", \"name\": \"id\"}, {\"type\": \"string\", \"name\": \"str\"}]}",
        "batch.size": "10000",
        "completed.dirname": "/g/g0/gimenez1/local/src/sonar/sonar-driver/mycompleteddir",
        "connector.class": "gov.llnl.sonar.kafka.connect.connectors.DirectorySourceConnector",
        "dirname": "/g/g0/gimenez1/local/src/sonar/sonar-driver/myingestdir",
        "format": "json",
        "format.options": "{}",
        "tasks.max": "1",
        "topic": "t1527303578.4373841-h-1048117146850943506"
    },
    "name": "sonar_directory_source-t1527303578.4373841-h-1048117146850943506"
}
```

The Avro schema file specifies the schema of data in the files (see [Creating Connector Avro Schemas](./creating_avro_schemas.md))
In this case each record in the files contains an integer named "id" and a string named "str".
You can create a Cassandra table with the same schema by using the `create_cassandra_table` script (see [Creating Cassandra Tables](./creating_cassandra_tables.md)).

See `create_directory_source_connector --help` for more details.

## Cassandra Sink Example

The following command:

```bash
create_cassandra_sink_connector -u theuser -p passfile mytopic mykey mytable
```

will produce:

```bash
{
    "config": {
        "connect.cassandra.contact.points": "localhost",
        "connect.cassandra.kcql": "INSERT INTO mytable SELECT * FROM mytopic",
        "connect.cassandra.key.space": "mykey",
        "connect.cassandra.password.file": "passfile",
        "connect.cassandra.port": "9042",
        "connect.cassandra.username": "theuser",
        "connector.class": "com.datamountaineer.streamreactor.connect.cassandra.sink.CassandraSinkConnector",
	"delete.sunk.kafka.records": "false",
        "tasks.max": "1",
        "topic": "mytopic"
    },
    "name": "cassandra_sink-mytopic"
}
```

See `create_cassandra_sink_connector --help` for more details.

Next - [Installing and Uninstalling Connectors](./installing_connectors.md)
