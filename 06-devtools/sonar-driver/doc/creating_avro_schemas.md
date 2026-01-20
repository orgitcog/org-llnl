# Creating Connector Avro Schemas

[Apache Avro](https://avro.apache.org/) is a specification for defining schemas.
Connectors use these schemas to parse data types from different sources into different sinks.
Sonar connectors all store information in records, which are typed tuples (as opposed to singular values). 
Therefore, all Avro schemas for Sonar connectors must define the types of different fields in each record.

Avro schemas are defined using JSON. For records, the top level type is always "record". 
Fields within the record simply have a name and an Avro data type.
For example, if we have records containing one integer called "id" followed by one string called "str", we would
define the following schema:

```JSON
{
    "type": "record",
    "name": "idstr",
    "fields": [
        {
            "type": "int",
            "name": "id"
        },
        {
            "type": "string",
            "name": "str"
        }
    ]
}
```

Note that the field ordering matters because it dictates the indices of the fields.

Avro schemas may be provided to Sonar connectors in two ways:

1. By creating an Avro schema file like the one above and specifying the filename in the connector 
configuration option "avro.schema.file", e.g. in the `avro_schema_file` parameter in the `create_directory_source_connector` script.
2. By providing the Avro as a string to the connector configuration option "avro.schema", taking care to 
escape quotes properly.

To properly escape quotes when defining the "avro.schema" option, for example using the all double quotes must be escaped, e.g.:


```
"{
    \"type\": \"record\",
    \"name\": \"idstr\",
    \"fields\": [
        {
            \"type\": \"int\",
            \"name\": \"id\"
        },
        {
            \"type\": \"string\",
            \"name\": \"str\"
        }
    ]
}"
```
