# Deploying

For Kafka Connect to use connectors, they must be provided in a location pointed to by the variable `plugin.path`
in the connector properties.
Within that location, we can either deploy a fat jar or a collection of jars (see [building](building.md))

Fat jars must reside in the `plugin.path` directly (not in a subdirectory), while collections of jars must reside
in a single subdirectory.
For example, if the plugin path is `/usr/share/kafka-plugins`, we could have:

```
/usr/share/kafka-plugins/fatJarConnector.jar
/usr/share/kafka-plugins/myConnector/myConnector.jar
/usr/share/kafka-plugins/myConnector/myConnectorDependency1.jar
/usr/share/kafka-plugins/myConnector/myConnectorDependency2.jar
```

If done incorrectly, dependency errors can occur. 
For more information, see [troubleshooting](troubleshooting.md).
