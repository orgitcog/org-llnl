# Building

Build using [Maven](https://maven.apache.org/).

Create the connectors jars from the top-level directory using:

```
mvn package
```

This creates the connector in two ways:
1. As a "fat jar" or "uber jar" containing all dependencies, in `target/uberJar`
2. As a regular jar with the Sonar connectors and all dependencies as separate jars, in `target/plugin/sonar-connectors`

In some cases it may be preferable to deploy using either the fat jar or the collection of jars, see [deployment](deploying.md).

That's it! Now [test it](testing.md).