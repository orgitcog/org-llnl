# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import re
import json
import os
from pathlib import Path
from loguru import logger
from zipfile import ZipFile
import tomli
import tomli_w
import copy
from collections import defaultdict
from typing import List, Dict, Any, Tuple, TypedDict

from ossp.database import get_asset_list, get_manufacturer_list, run_sql_script, get_view_columns, get_view_values
from ossp.redaction.data_converters import table_to_json, json_to_csv


DATABASE = Path("/data/databases/database.db")
DATABASE_VIEWS = ["AssetDataExport", "FirmwareSummaryExport", "OrgExport", "ProgramsExport", "SubComponentExport"]

class PolicyDict(TypedDict, total=False):
    redactkeys: List[str]
    redactvalues: List[Tuple[str, str]]
    redactrecords: List[Tuple[str, str]]
    redactvaluesforkey: List[Tuple[str, str, str]]

# ------------------------------------Create Redaction Toml---------------------------------------#

def create_column_spec():
    """
    Get a list of the columns in the Database views.
    """
    tables = []

    for view in DATABASE_VIEWS:
        columns = get_view_columns(DATABASE,view)
        entry = {
                "name": view,
                "columns": [{"name": col} for col in columns],
            }
        tables.append(entry)

    return tables

def create_view_values():
    """
    Get a list of the values for each column within the Database views.
    """
    tables = []

    for view in DATABASE_VIEWS:
        columns = get_view_values(DATABASE,view)
        entry = {
                "name": view,
                "values": [{"name": col} for col in columns],
            }
        tables.append(entry)

    return tables

def build_policy_from_rules(rules: List[Dict[str, Any]]) -> PolicyDict:
    """
    Convert the frontend rules array into four lists:
      - redactkeys           : List[str]
      - redactvalues         : List[str]        # just patterns
      - redactrecords        : List[str]        # just patterns
      - redactvaluesforkey   : List[List[str]]  # each [key, pattern]
    """
    redactkeys:          List[str]       = []
    redactvalues:        List[Tuple[str,str]]       = []
    redactrecords:       List[Tuple[str,str]]       = []
    redactvaluesforkey:  List[Tuple[str,str,str]] = []

    for rule in rules:
        t      = rule.get("type")
        key    = rule.get("key")     # may be None for byKeyName
        values = rule.get("values", [])

        if t == "byKeyName":
            redactkeys.extend(values)

        elif t == "byFieldValue":
            # key="Brand", values=["Cisco","GE.*"]
            # pattern = [ ["Brand","Cisco"], ["Brand","GE.*"], … ]
            if not key:
                raise ValueError("byFieldValue rule missing 'key'")
            for pattern in values:
                redactvalues.append((key, pattern))

        elif t == "rowByFieldValue":
            # key="Model", values=["Other.*",…]
            # pattern = [ ["Model","Other.*"], … ]
            if not key:
                raise ValueError("rowByFieldValue rule missing 'key'")
            for pattern in values:
                redactrecords.append((key, pattern))

        elif t == "multiKeyValue":
            # values = [ [triggerKey,triggerPattern,"field1,field2"], … ]
            for entry in values:
                if not (isinstance(entry, list) and len(entry) == 3):
                    raise ValueError(f"Invalid multiKeyValue entry: {entry!r}")
                key, pattern, fields = entry
                redactvaluesforkey.append((key, pattern, fields))

        else:
            raise ValueError(f"Unknown rule type: {t!r}")

    policy: PolicyDict = {}
    if redactkeys:
        policy["redactkeys"] = redactkeys
    if redactvalues:
        policy["redactvalues"] = redactvalues
    if redactrecords:
        policy["redactrecords"] = redactrecords
    if redactvaluesforkey:
        policy["redactvaluesforkey"] = redactvaluesforkey

    return policy

def policy_to_rules(toml_file: Path, policy_name: str) -> List[Dict[str, Any]]:
    """
    Convert a TOML policy into the frontend rules array:
      - byKeyName:        {"type":"byKeyName", "key":None, "values":[...]}
      - byFieldValue:     {"type":"byFieldValue", "key":k, "values":[pattern1, ...]}
      - rowByFieldValue:  {"type":"rowByFieldValue", "key":k, "values":[pattern1, ...]}
      - multiKeyValue:    {"type":"multiKeyValue", "key":k, "values":[[k, pattern, "f1,f2"], ...]}
    """
    redactkeys, redactvalues, redactrecords, redactvaluesforkey = get_policy(toml_file, policy_name)

    rules: List[Dict[str, Any]] = []

    if redactkeys:
        rules.append({"type": "byKeyName", "key": None, "values": list(redactkeys)})

    for k, patterns in redactvalues.items():
        if patterns:
            rules.append({"type": "byFieldValue", "key": k, "values": list(patterns)})

    for k, patterns in redactrecords.items():
        if patterns:
            rules.append({"type": "rowByFieldValue", "key": k, "values": list(patterns)})

    for k, pairs in redactvaluesforkey.items():
        if pairs:
            values = [[k, patt, fields] for (patt, fields) in pairs]
            rules.append({"type": "multiKeyValue", "key": k, "values": values})

    return rules

def upsert_policy(toml_file: Path, policy_name: str, policy_dict: Dict[str, Any]) -> None:
    """
    Merge or add a policy under [policy_name] in redact.toml without removing other tables.
    """
    config: Dict[str, Any] = {}
    if toml_file.exists():
        with open(toml_file, "rb") as fp:
            config = tomli.load(fp)

    config[policy_name] = policy_dict

    toml_text = tomli_w.dumps(config)
    toml_file.parent.mkdir(parents=True, exist_ok=True)
    with open(toml_file, "w", encoding="utf-8") as fp:
        fp.write(toml_text)

def normalize_policy_dict(policy_dict: PolicyDict) -> Tuple[
    List[str], 
    Dict[str, List[str]], 
    Dict[str, List[str]], 
    Dict[str, List[Tuple[str,str]]]]:
    """
    Turn the dict returned by build_policy_from_rules into the 4-tuple
    (redactkeys, redactvalues, redactrecords, redactvaluesforkey)
    that redact_by_policy expects.
    """
    redactkeys = policy_dict.get("redactkeys", [])

    rv: Dict[str, List[str]] = {}
    for k, patt in policy_dict.get("redactvalues", []):
        rv.setdefault(k, []).append(patt)

    rr: Dict[str, List[str]] = {}
    for k, patt in policy_dict.get("redactrecords", []):
        rr.setdefault(k, []).append(patt)

    rvfk: Dict[str, List[Tuple[str, str]]] = {}
    for k, patt, fields in policy_dict.get("redactvaluesforkey", []):
        rvfk.setdefault(k, []).append((patt, fields))

    return redactkeys, rv, rr, rvfk

# ------------------------------------Redaction---------------------------------------#
def _stringify_rows(rows: list[dict]) -> list[dict]:
    """ 
    Return a deep copy of rows with all values as strings. None becomes ''.
    """
    out = []
    for row in rows:
        newrow = {}
        for k, v in row.items():
            if v is None:
                newrow[k] = ""
            elif isinstance(v, str):
                newrow[k] = v
            else:
                newrow[k] = str(v)
        out.append(newrow)
    return out

def redact(
    dbname: Path,
    redacted_folder: Path,
    toml_file: Path,
    policy_name: str = "None",
    policy: tuple | None = None,
):

    json_data: dict = {
        "FirmwareSummaryExport": [],
        "AssetDataExport": [],
        "ProgramsExport": [],
        "OrgExport": [],
        "SubComponentExport": [],
    }
    if policy is None:
        policy = get_policy(toml_file, policy_name)

    logger.info(f"Policy selected = {policy_name}")

    for table in json_data.keys():
        rows = table_to_json(dbname, table)
        rows = _stringify_rows(rows)
        json_data[table] = redact_by_policy(rows, policy)

    if not redacted_folder.exists():
        redacted_folder.mkdir()

    # manifest to keep track of the redaction rules associated with a zip file
    manifest = {
        "policy_name": policy_name,
        "redactkeys": list(policy[0]),
        "redactvalues": {k: list(v) for k, v in policy[1].items()},
        "redactrecords": {k: list(v) for k, v in policy[2].items()},
        "redactvaluesforkey": {k: v for k, v in policy[3].items()},  # list of [pattern, fields]
    }

    with ZipFile(redacted_folder.with_suffix(".zip"), "w") as redacted_zip:
        for table, redacted_data in json_data.items():
            filename = redacted_folder / f"{table}.csv"
            json_to_csv(redacted_data, filename)
            redacted_zip.write(filename, arcname=filename.name)
        redacted_zip.writestr("policy_manifest.json", json.dumps(manifest, indent=2))


def redact_by_policy(json_data: list | dict, policy: tuple):
    if len(policy) != 4:
        raise RuntimeError(
            f"Policy is not correctly formatted, please review toml config file."
        )
    redactkeys, redactvalues, redactrecords, redactvaluesforkey = policy
    row = 0
    new_json_data = []
    # Loop through every row in Json
    while row < len(json_data):
        newrow = copy.deepcopy(json_data[row])
        storerow = 1
        for key in newrow:
            #
            # Redact Full Record Based on Match
            #
            for matchkey in redactrecords:
                if matchkey == key:
                    for value in redactrecords[matchkey]:  # Loop through list
                        pattern = re.compile(value.lower())
                        tomatch = json_data[row][key].lower()
                        if pattern.match(tomatch):
                            storerow = 0
                            newrow = ""
                            break
            if storerow == 0:
                break
            #
            # Redact Values for Specific Key Names
            #
            if key.lower() in map(str.lower, redactkeys):
                newrow[key] = ""
            #
            # Redact Value based on Value for Key
            #
            for matchkey in redactvalues:
                if matchkey == key:
                    for value in redactvalues[matchkey]:  # Loop through list
                        pattern = re.compile(value.lower())
                        if pattern.match(json_data[row][key].lower()):
                            newrow[key] = ""
                        #
                        # Redact multiple Values based on Value for Key
                        #
            for matchkey in redactvaluesforkey:
                if matchkey == key:
                    for pair in redactvaluesforkey[matchkey]:  # loop through list
                        pattern = re.compile(pair[0].lower())
                        if pattern.match(json_data[row][key].lower()):
                            tmplist = pair[1].split(",")
                            for entry in tmplist:
                                if entry in newrow:
                                    newrow[entry] = ""

        if storerow > 0:
            new_json_data.append(newrow)

        row += 1
    return new_json_data


# Adapted from code written by Craig Gorenson from PNNL
def get_policy(toml_file: Path, policy_name: str):

    with open(toml_file, mode="rb") as fp:
        config = tomli.load(fp)

    if policy_name not in config.keys():
        logger.error(f"No policy called {policy_name} exists")
        if "default" not in config.keys():
            raise KeyError(
                "No matching policy found, please provide policy specification."
            )
        else:
            policy_name = "default"

    redactkeys = []
    if "redactkeys" in config[policy_name]:
        redactkeys = config[policy_name]["redactkeys"]
    # Built Redaction Value table from Dict
    # Toml doesn't store dict, so stored dict at nested array
    # then converting it to Dict on Load
    redactvalues = defaultdict(list)
    if "redactvalues" in config[policy_name]:
        tmp_redactvalues = config[policy_name]["redactvalues"]
        for v in tmp_redactvalues:
            redactvalues[v[0]].append(v[1])
        logger.debug("Redact Values: " + str(redactvalues))

    # Built Redaction records table from Dict
    # Toml doesn't store dict, so stored dict at nested array
    # then converting it to Dict on Load
    redactrecords = defaultdict(list)
    if "redactrecords" in config[policy_name]:
        tmp_redactrecords = config[policy_name]["redactrecords"]
        for v in tmp_redactrecords:
            redactrecords[v[0]].append(v[1])
        logger.debug("Redact Records:" + str(redactrecords))

    # Built Redaction records table from Dict
    # Toml doesn't store dict, so stored dict at nested array
    # then converting it to Dict on Load
    redactvaluesforkey = defaultdict(list)
    if "redactvaluesforkey" in config[policy_name]:
        tmp_valuesforkey = config[policy_name]["redactvaluesforkey"]
        for v in tmp_valuesforkey:
            redactvaluesforkey[v[0]].append([v[1], v[2]])
        logger.debug("Redact ValuesForkeys:" + str(redactvaluesforkey))

    return redactkeys, redactvalues, redactrecords, redactvaluesforkey
