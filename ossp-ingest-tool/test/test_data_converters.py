from ossp.redaction.data_converters import table_to_json, json_to_csv
from ossp.database import run_sql_script

import tempfile
import pandas as pd

SCHEMA = """
BEGIN TRANSACTION;
CREATE TABLE IF NOT EXISTS "FirmwareSummaryExport" (
	"ComponentID"	INTEGER,
	"Component"	TEXT,
	PRIMARY KEY("ComponentID")
);
CREATE TABLE IF NOT EXISTS "AssetData" (
	"AssetID"	INTEGER NOT NULL,
	"AssetName"	TEXT,
	PRIMARY KEY("AssetID")
);
COMMIT;
"""
DATA = """
BEGIN TRANSACTION;
INSERT INTO FirmwareSummaryExport (ComponentID, Component) 
VALUES ('1', 'openssl'),
       ('2', 'openssh'),
       ('3', 'libc'),
       ('4', 'glibc');
INSERT INTO AssetData (AssetID, AssetName) 
VALUES ('1', 'Modicon M221'),
       ('2', 'SEL 310'),
       ('3', 'Hometroller 67'),
       ('4', 'TI 898');
COMMIT;
"""


def setup_temp_db(dbname):
    run_sql_script(dbname, SCHEMA)
    run_sql_script(dbname, DATA)


def test_table_to_json():
    expected_json = [
        {"ComponentID": 1, "Component": "openssl"},
        {"ComponentID": 2, "Component": "openssh"},
        {"ComponentID": 3, "Component": "libc"},
        {"ComponentID": 4, "Component": "glibc"},
    ]

    with tempfile.NamedTemporaryFile(mode="w+t", delete=True) as dbname:
        setup_temp_db(dbname.name)
        json = table_to_json(dbname.name, "FirmwareSummaryExport")
    assert json == expected_json


def test_json_to_csv():
    json = {
        "ComponentID": {0: 1, 1: 2, 2: 3, 3: 4},
        "Component": {0: "openssl", 1: "openssh", 2: "libc", 3: "glibc"},
    }
    with tempfile.NamedTemporaryFile(mode="w+t", delete=True) as csv_file:
        success = json_to_csv(json, csv_file.name)
        assert success
        df = pd.read_csv(csv_file.name)
        assert df["Component"][1] == "openssh"
