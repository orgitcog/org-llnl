# Cassandra on Sonar

`sonar-driver` provides the `SonarCassandraSession` class to create an authenticated session with 
Sonar's Cassandra database. 

Right now there are two use cases: connecting directly to the Cassandra database with the python Cassandra driver and 
connecting to the database via Spark.
In either case, you must first initialize a Sonar Cassandra session.

## Creating a Sonar Cassandra session

Initializing a `SonarCassandraSession` instance will prompt you for your password interactively (works in Jupyter as well).

```python
from sonar_driver.cassandra.session import SonarCassandraSession
scs = SonarCassandraSession(['sonar8'])
```

Upon successful password entry, SonarCassandraSession will initialize a persistent Cassandra session.

### Cassandra Driver Connection

After creating a SonarCassandraSession as above, simply access the Python Cassandra driver session via:

```python
scs.session
```

See [Python Cassandra driver documentation](https://datastax.github.io/python-driver/) for usage.

### Spark Cassandra Connection

After creating a SonarCassandraSession, create a SonarSparkSession, passing the Cassandra session in 
the constructor (see [Spark on Sonar](spark.md)).


