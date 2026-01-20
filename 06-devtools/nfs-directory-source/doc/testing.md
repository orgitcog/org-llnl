# Testing

Install confluent and add it to PATH, e.g. `PATH=~/confluent-4.0.0/bin:$PATH`

Add the packaged connectors to confluent's schema-registry properties, e.g.:

In files:
```
confluent-4.0.0/etc/schema-registry/connect-avro-standalone.properties
confluent-4.0.0/etc/schema-registry/connect-avro-standalone.properties
```

Modify the last line from the default:

```
plugin.path=share/java
```

...to include the absolute location of the packaged connectors jar file we built in `./target`:

```
plugin.path=share/java,~/src/kafkaconnectors/target
```

Now you should be able to run `bin/test.sh`

Successful test output looks like:

```
========== INITIALIZING TEST STATE ===========
Clearing out confluent...
Stopping connect
connect is [DOWN]
Stopping kafka-rest
kafka-rest is [DOWN]
Stopping schema-registry
schema-registry is [DOWN]
Stopping kafka
kafka is [DOWN]
Stopping zookeeper
zookeeper is [DOWN]
Deleting: /var/folders/p8/vbk8nbcd5n594hgbc89xpnh4001_x9/T/confluent.WB5DDNTw
Starting kafka connect (and dependencies)...
Starting zookeeper
zookeeper is [UP]
Starting kafka
kafka is [UP]
Starting schema-registry
schema-registry is [UP]
Starting kafka-rest
kafka-rest is [UP]
Starting connect
connect is [UP]
Testing whether LLNLFileSourceConnector is available...found!
========== RUNNING TESTS ===========
Loading test connector: test_idstr
{
  "name": "test_idstr",
  "config": {
    "tasks.max": "1",
    "connector.class": "LLNLFileSourceConnector",
    "filename": "../src/test/resources/test_idstr.json",
    "format": "json",
    "format.options": "",
    "topic": "test_idstr_topic",
    "avro.schema.filename": "../src/test/resources/test_idstr.avsc",
    "name": "test_idstr"
  },
  "tasks": [],
  "type": null
}
Validating...PASS
Loading test connector: test_alltypes
{
  "name": "test_alltypes",
  "config": {
    "tasks.max": "1",
    "connector.class": "LLNLFileSourceConnector",
    "filename": "../src/test/resources/test_alltypes.json",
    "format": "json",
    "format.options": "",
    "topic": "test_alltypes_topic",
    "avro.schema.filename": "../src/test/resources/test_alltypes.avsc",
    "name": "test_alltypes"
  },
  "tasks": [],
  "type": null
}
Validating...PASS
========== TEST RESULTS ===========
Tests passed: 2
Tests failed: 0
```