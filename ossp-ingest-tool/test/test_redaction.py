from ossp.redaction.redact import get_policy, redact_by_policy

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


# def setup_temp_db(dbname):
#     run_sql_script(dbname, SCHEMA)
#     run_sql_script(dbname, DATA)


def test_get_policy():
    policy = """
        [default]
        redactkeys = [ 'Name', 'Names', 'Make','FirmwareVersion' ]
        redactvalues = [ ["Brand","Cisco.*"], ["Model","SEL.*"] ]
        redactrecords = [ ["Brand","GE.*"], ["Model","Other.*"] ]
        redactvaluesforkey = [ [ "Name","Arbiter.1133A.*","Make,Model,FirmwareVersion,Site"], ["Make","This.*","Make,Model,FirmwareVersion"] ]

        [assetData]
        redactkeys = [ 'Name', 'Names', 'Make','FirmwareVersion' ]
        redactvalues = [ ["Brand","Cisco.*"], ["Model","SEL.*"] ]
    """
    with tempfile.NamedTemporaryFile(mode="w") as policy_file:
        policy_file.write(policy)
        policy_file.seek(0)
        redactkeys, redactvalues, redactrecords, redactvaluesforkey = get_policy(
            policy_file.name, "default"
        )

    assert redactkeys == ["Name", "Names", "Make", "FirmwareVersion"]
    assert redactvalues["Brand"] == ["Cisco.*"]
    assert redactvalues["Model"] == ["SEL.*"]
    assert redactrecords["Brand"] == ["GE.*"]
    assert redactrecords["Model"] == ["Other.*"]
    assert redactvaluesforkey["Name"] == [
        ["Arbiter.1133A.*", "Make,Model,FirmwareVersion,Site"]
    ]
    assert redactvaluesforkey["Make"] == [["This.*", "Make,Model,FirmwareVersion"]]


def test_redact_by_keys():
    data = [
        {"ComponentID": "1", "Component": "openssl", "MAC": "3243"},
        {"ComponentID": "2", "Component": "openssh", "MAC": "5345"},
        {"ComponentID": "3", "Component": "libc", "MAC": "0923"},
        {"ComponentID": "4", "Component": "glibc", "MAC": "3342"},
    ]
    policy = ["MAC"], {}, {}, {}
    redacted_data = redact_by_policy(data, policy)

    assert len(redacted_data) == 4
    result = set([entry["MAC"] for entry in redacted_data])
    assert len(result) == 1
    assert result == set([""])


def test_redact_by_values():
    data = [
        {"ComponentID": "1", "Component": "openssl", "MAC": "3243"},
        {"ComponentID": "2", "Component": "openssh", "MAC": "5345"},
        {"ComponentID": "3", "Component": "libc", "MAC": "0923"},
        {"ComponentID": "4", "Component": "glibc", "MAC": "3342"},
    ]
    policy = [], {"Component": ["open.*"]}, {}, {}
    redacted_data = redact_by_policy(data, policy)

    assert len(redacted_data) == 4
    assert redacted_data[1]["Component"] == ""
    assert redacted_data[0]["Component"] == ""


def test_redact_by_records():
    data = [
        {"ComponentID": "1", "Component": "openssl", "MAC": "3243"},
        {"ComponentID": "2", "Component": "openssh", "MAC": "5345"},
        {"ComponentID": "3", "Component": "libc", "MAC": "0923"},
        {"ComponentID": "4", "Component": "glibc", "MAC": "3342"},
    ]
    policy = [], {}, {"Component": ["open.*"]}, {}
    redacted_data = redact_by_policy(data, policy)

    assert len(redacted_data) == 2
    assert {
        "ComponentID": "1",
        "Component": "openssl",
        "MAC": "3243",
    } not in redacted_data
    assert {"ComponentID": "3", "Component": "libc", "MAC": "0923"} in redacted_data


def test_redact_by_records_same_key():
    data = [
        {"ComponentID": "1", "Component": "openssl", "MAC": "3243"},
        {"ComponentID": "2", "Component": "openssh", "MAC": "5345"},
        {"ComponentID": "3", "Component": "libc", "MAC": "0923"},
        {"ComponentID": "4", "Component": "glibc", "MAC": "3342"},
    ]
    policy = [], {}, {"Component": ["open.*", "libc"]}, {}
    redacted_data = redact_by_policy(data, policy)

    assert redacted_data == [{"ComponentID": "4", "Component": "glibc", "MAC": "3342"}]


def test_redact_by_valuesforkey():
    data = [
        {"ComponentID": "1", "Component": "openssl", "MAC": "3243"},
        {"ComponentID": "2", "Component": "openssh", "MAC": "5345"},
        {"ComponentID": "3", "Component": "libc", "MAC": "0923"},
        {"ComponentID": "4", "Component": "glibc", "MAC": "3342"},
    ]
    redactvaluesforkey = {"MAC": [["3243", "Component"]], "ComponentID": [["2", "MAC"]]}
    policy = [], {}, {}, redactvaluesforkey
    redacted_data = redact_by_policy(data, policy)

    assert len(redacted_data) == 4
    assert redacted_data[0]["Component"] == ""
    assert redacted_data[1]["MAC"] == ""


def test_redact_by_valuesforkey_same_key():
    data = [
        {"ComponentID": "1", "Component": "openssl", "MAC": "3243"},
        {"ComponentID": "2", "Component": "openssh", "MAC": "5345"},
        {"ComponentID": "3", "Component": "libc", "MAC": "0923"},
        {"ComponentID": "4", "Component": "glibc", "MAC": "3342"},
    ]
    redactvaluesforkey = {
        "MAC": [["3243", "Component"], ["3342", "ComponentID"]],
        "ComponentID": [["2", "MAC"]],
    }
    policy = [], {}, {}, redactvaluesforkey
    redacted_data = redact_by_policy(data, policy)

    assert len(redacted_data) == 4
    assert redacted_data[0]["Component"] == ""
    assert redacted_data[1]["MAC"] == ""
    assert redacted_data[3]["ComponentID"] == ""
