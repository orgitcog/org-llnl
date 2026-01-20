# Creating Cassandra Tables

The `create_cassandra_table` script provides a simple mechanism to create a Cassandra table from an Avro schema.
Invoke with `--dry/-d` or `--debug/-g` to see the ensuing CQL commands.

## Example

The command:

```bash
create_cassandra_table -u theuser -p passfile -pk id mykeyspace mytable ./tests/idstr.avsc
```

will create a table with the equivalent Cassandra schema as specified in the Avro schema file `./tests/idstr.avsc` and "id" as the primary key, using the CQL command:

```bash
CREATE TABLE mykeyspace.mytable (id int,str text, PRIMARY KEY ((id)))
```

You can also specify a clustering key via `--cluster-key/-ck`.

If the table already exists, the command will fail.
