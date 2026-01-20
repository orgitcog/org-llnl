# Spark on Sonar

`sonar-driver` provides the `SonarSparkSession` class for initializing a Spark session 
on Sonar, optionally with support for connecting Spark to Cassandra.

## Creating a Sonar Spark session

Use the following to construct a Sonar Spark session:

```python
from sonar_driver.spark.session import SonarSparkSession

sonar_spark = SonarSparkSession(spark_home='path/to/spark',
                 sonar_cassandra_session=None,                          # default
                 spark_opts={'master': 'local[*]'},                     # default
                 java_home='/usr/lib/jvm/java-1.8.0/',                  # default
                 trust_store='/etc/pki/ca-trust/extracted/java/cacerts' # default
             )
```

### Parameters

* `spark_home`: Spark home directory
* `sonar_cassandra_session`: (optional) A `SonarCassandraSession` instance with an authenticated connection to Cassandra 
(see [Cassandra on Sonar](cassandra.md))
* `spark_opts`: Options for submitting Spark jobs (see 
[Spark configuration](https://spark.apache.org/docs/latest/configuration.html))
* `java_home`: Java home directory (what environment variable `JAVA_HOME` should be set to)
* `trust_store`: Location of Java trust store (used for downloading Spark packages)

Defaults are configured for Sonar/RZSonar.

## Using Spark

A `SonarSparkSession` instance contains the `spark_session` member, an instantiated instance of 
`org.apache.spark.sql.SparkSession`, which can be used directly (see 
[Spark Dataframe API](https://spark.apache.org/docs/latest/sql-programming-guide.html#dataframes)) or used to access an instantiated 
`org.apache.spark.SparkContext` via `spark_session.sparkContext` (see 
[Spark RDD API](https://spark.apache.org/docs/latest/rdd-programming-guide.html#resilient-distributed-datasets-rdds)).

### Loading a Cassandra Table

`SonarSparkSession` downloads the [spark-cassandra-connector](https://github.com/datastax/spark-cassandra-connector) 
package (see [documentation for using spark-cassandra-connector in Python](https://github.com/datastax/spark-cassandra-connector/blob/master/doc/15_python.md)).

Load a Cassandra table like so:

```python
df = sonar_spark.spark_session.format("org.apache.spark.sql.cassandra") \
    .options(keyspace="keyspace", table="table") \
    .load()
```
