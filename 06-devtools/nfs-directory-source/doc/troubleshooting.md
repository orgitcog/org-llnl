# Troubleshooting

If you encounter an "ERROR" in the Kafka connect logs, you're not alone! 
Here is a list of commonly occurring errors and solution strategies:

## NoClassDefFound 

For "reasons", sometimes dependency errors occur even when correctly deploying connector jars, resulting in Kafka 
connect logs  producing `NoClassDefFound` exceptions or similar.
This is only a problem if an ERROR message is logged--upon startup, lots of these exceptions may pop up as WARN 
(warnings) from the Java Reflections API. Don't worry about those.

1. In practice, on Sonar, fat jars are preferable, and on local testing systems (i.e. your laptop), collections of jars
are preferable. Try switching between both kinds of deployments.

2. If switching the deployment type doesn't resolve the issue, you can check whether the dependencies are present using
the `jar` tool (in Java's path).
For example, if `NoClassDefFound` references `org/json/JSONObject` and the deployment uses a fat jar 
`/usr/share/kafka-plugins/fatJar.jar`, then run:

```bash
jar tf /usr/share/kafka-plugins/fatJar.jar | grep JSONObject
```

If using a collection of jars, e.g., in `/usr/share/kafka-plugins/myConnector`, use `find` to inspect all jars:


```bash
find /usr/share/kafka-plugins/myConnector -name "*.jar" -exec jar tf {} \; | grep JSONObject
```

If the dependency is not there, you may need to rebuild and/or redeploy the connectors.
If it is, check the deployment, particularly the `plugin.path` in the connector properties and the file structure 
within the path.

## ZooKeeper CONNECTED over and over

A ZooKeeper connection is kept alive by sending "heartbeat" signals
from the client to the server. 
After a certain length of time passes between heartbeats, a ZooKeeper connection is closed.
The Sonar directory source connector uses ZooKeeper through the Apache Curator API, and its timeout is set to be really 
really long for continuous ingestion (connector tasks close the connection explicitly when finished).
However, some situations can cause the connection to timeout, in which case Curator attempts to reconnect
over and over again, hence the log messages. 
At this point, typically the connector becomes completely dysfunctional (and logs get bloated).

Typically, this occurs because the Sonar connector filled up ZooKeeper with too much crap--er, file offset data.

1. Stop the directory source connector causing the CONNECTED logs
2. Purge the file offsets stored in ZooKeeper. ZooKeeper looks like a filesystem, and the ZooKeeper offsets for a 
directory source `/some/dir/source` are located in an identical ZooKeeper path `/some/dir/source`. Delete it using the 
`zookeeper-client` provided by `confluent`.

```bash
zookeeper-client rzsonar8:2181 rm /some/dir/source
```

3. Restart the connector, and it will repopulate the file offsets (potentially re-ingesting data if it was not purged
from the directory source location).

## NullPointerException

Typically this indicates poor connector configuration. Find the first ERROR log before this, as it may provide
some more information. Otherwise, make sure all required connector configurations are specified.

## ValueError: Expecting value: line 1 column 1 (char 0)

You used the `nfs-directory-source` to do something and got this weird JSON error. 
This means the server responded with an error status code, rather than a useful JSON message. 
Check the confluent connect logs for something useful.

## avro.schema.SchemaParseException: Invalid schema name

Avro has some weird/annoying restrictions. You can validate your schema with an Avro schema validator 
[like this one](https://json-schema-validator.herokuapp.com/avro.jsp) and read further documentation on the restrictions
in [Avro's documentation](https://avro.apache.org/docs/current/).

## org.apache.kafka.connect.errors.DataException: Found null value for non-optional schema

Your config does not match the data.
Config properties that may be at fault:
- "avro.schema" the schema has incorrect fields
- "format.options" for a CSV, the "columns" within "format.options" is incorrect or undefined (and there is no CSV header)

Otherwise, there may be a null value in the data that is not specified as nullable by the Avro schema.
You may alter the schema to allow nullable for each field.

## java.lang.OutOfMemoryError: Java heap space

Your batches are too large and/or your number of tasks is too high!
Try specifying a smaller "batch.rows" and/or "batch.files" and/or "tasks.max" in the connector config.
If you're using `create_directory_source_connector`, you can use the `--batch-rows`/`--batch-files`/`--tasks-max` options.
The total batch size is the product of rows*files.