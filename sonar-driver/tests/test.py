import json
import avro.schema

from sonar_driver.print_utils import pretty_print
from sonar_driver.connector import Connector
from sonar_driver.sonar_directory_source_config import SonarDirectorySourceConfig

avro_schema_json = {
    "name": "myrecord",
    "type": "record",
    "fields": [
        {
            "name": "id",
            "type": "int"
        },
        {
            "name": "str",
            "type": "string"
        }
    ]
}

c = Connector(
    "myconnector",
    SonarDirectorySourceConfig(
        "mytopic",
        "mydirname",
        "mycompleteddirname",
        avro.schema.Parse(json.dumps(avro_schema_json))
    )
)

pretty_print("myconnector", c.json())
