# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import pandas as pd
from loguru import logger
from pathlib import Path
import csv
import re
import sqlite3

from ossp.database import *
from ossp.database import (
    select_component_id,
    insert_into_Programs,
)  # Needed to add this because mypy kept having issues
from ossp.process_sboms import get_excluded_sboms

csv.field_size_limit(10 * 1024*1024)  # Increase field size limit to 1MB

def get_sheet_from_excel(filename, sheetname):
    """Reads in sheet from an excel file into a Pandas DataFrame."""
    try:
        df = pd.read_excel(filename, sheet_name=sheetname)
        return df
    except Exception as e:
        logger.error(f"Failed to read excel sheet {sheetname} from {filename} - {e}")
        raise FileNotFoundError("Asset data not found, please upload the asset list.")


# Code adapted from Craig Goransen at PNNL
def _split_string(delimited_text):
    """Splits license string into a comma delimited list."""
    delimiter = ","
    if isinstance(delimited_text, str):
        # Convert OR,and,AND to comma and remove 'withs'
        text = (
            delimited_text.replace(" with exceptions", "")
            .replace(" with advertising", "")
            .replace(" and additional rights", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" and ", ",")
            .replace(" AND ", ",")
            .replace(" AND", ",")
            .replace(" or ", ",")
            .replace(" OR ", ",")
            .replace("/", ",")
        )
        if "ISC+IBM" in text:
            text = text.replace("ISC+IBM", "ISC,IBM")
        if "MIT-X11" in text:
            text = text.replace("MIT-X11", "MIT,X11")
        if "MPL-2.0 GPL-2.0-or-later" in text:
            text = text.replace("MPL-2.0 GPL-2.0-or-later", "MPL-2.0,GPL-2.0-or-later")
        if "MIT BSD GPL2+" in text:
            text = text.replace("MIT BSD GPL2+", "MIT,BSD,GPL2+")
        if text.count(delimiter) > 0:
            list = text.split(delimiter)
            length = len(list)
            for i in range(length):
                list[i] = list[i].strip()
            list.sort()
            text = ",".join(list)
        else:
            text = text.strip()
        return text
    else:
        return ""


def ingest_running_components(
    dbname: Path, assetid: str, sbomid: str, running_components: pd.DataFrame
):
    success = True
    for _, row in running_components.iterrows():
        component_id = select_component_id(
            dbname, sbomid, row["Component"], row["License"], row["Version"]
        )
        if component_id:
            success = (
                insert_into_Programs(
                    dbname, assetid, row["Component"], component_id, True
                )
                and success
            )
    return success


def preprocess_sbom_info(fullpath, firmwareid):
    """Prepares sbom version info for insertion into the FirmwareBomVer table."""
    filename = os.path.basename(fullpath)
    # Add check to make sure the file path exists
    if not Path(fullpath).exists():
        logger.warning(f"SBOM info csv not found: {fullpath}")
        return None
    
    df = pd.DataFrame()
    # Read CSV into pandas Dataframe
    try:
        df = pd.read_csv(fullpath, header=None, dtype="string")
    except pd.errors.EmptyDataError:
        logger.warning(f"{filename} contains no data")

    try:
        df.columns = [
            "BomSerialNum",
            "BomTimeStamp",
            "BomVersion",
            "BomRef",
            "FileName",
            "ComponentVersion",
            "FileType",
            "ToolName",
            "ToolVendor",
            "ToolVersion",
        ]
    except ValueError as e:
        logger.error(f"Pandas Error - number of columns not as expected - {e}")
        return None
    # Add the FirmwareID and BomFileName to the df
    df.insert(0, "FirmwareID", firmwareid)
    df.insert(1, "BomFileName", filename[:-7] + "json")
    return df


def preprocess_component_info(
    fullpath: Path, firmwareid: str, sbomid: str
) -> pd.DataFrame:
    df = pd.DataFrame()
    # Read CSV into pandas Dataframe
    try:
        df = pd.read_csv(fullpath, header=None, dtype="string")
    except pd.errors.EmptyDataError:
        logger.warning(f"{fullpath.name} contains no data")
        return pd.DataFrame()
    # Rename columns to match schema and add firmwareid
    try:
        df.columns = pd.Index(
            [
                "Component",
                "Version",
                "License",
                "ExternalLicenseRef",
                "ComponentType",
                "copyright",
                "CPE",
                "PURL",
                "BomRef",
                "description",
                "copyright2",
                "Author",
                "Running",
            ]
        )
        df.drop(["copyright", "copyright2", "description"], axis=1, inplace=True)
        df["FirmwareID"] = firmwareid
        df["SBOMID"] = sbomid
    except ValueError as e:
        logger.error(f"Pandas Error - number of columns not as expected - {e}")
        return pd.DataFrame()
    # Split license strings
    try:
        df["License"] = df["License"].apply(_split_string)
    except ValueError as e:
        logger.error(f"Pandas Error - could not apply license string split - {e}")
        return pd.DataFrame()
    # Remove component types that are not files
    try:
        df = df[df.ComponentType != "file"]
    except ValueError as e:
        logger.error(f"Pandas Error - could remove component types = file - {e}")
        return pd.DataFrame()

    return df


def preprocess_dep_file(fullpath, sbomid, firmwareid, dbname):
    """Prepares sbom dependency data for insertion into the SubComponents table."""
    # Open the csv file into a csv reader and read it line by line
    try:
        with open(fullpath, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row:
                    continue
                bom_refs = [ref for bom in row for ref in bom.split(",")]
                # The second element in this list returns a comma separated string of dependencies that need to be split
                Child_component = bom_refs[0]
                # Code adapted from Craig Goransen at PNNL
                depdict = get_deps_from_FirmwareSummary(
                    dbname, bom_refs, firmwareid, sbomid
                )
                rootnode = 1
                if len(bom_refs) == 1:
                    # There are no dependencies for this component. TODO: Add check in graph if the ParentID == ChildID otherwise it'll show a circular dependency.
                    ParentID = depdict[Child_component]
                    ChildID = depdict[Child_component]
                    insert_into_SubComponents(
                        dbname, ParentID, ChildID, rootnode, sbomid
                    )
                else:
                    for item in bom_refs[
                        1:
                    ]:  # Assuming the first element is the child component that depends on the others in the list
                        bomref2 = item
                        if Child_component in depdict and bomref2 in depdict:
                            ParentID = depdict[bomref2]
                            ChildID = depdict[Child_component]
                            insert_into_SubComponents(
                                dbname, ParentID, ChildID, rootnode, sbomid
                            )
    except pd.errors.EmptyDataError:
        logger.warning(f"{fullpath.name} contains no data")
        return False

    return True

def _base_without_csv(name: str) -> str:
    """returns the filename base without the csv ending"""
    return name[:-4] if name.lower().endswith(".csv") else name

def _ver_csv_path(p: Path) -> Path:
    """returns the filename base with the .ver.csv ending"""
    base = _base_without_csv(p.name)
    return p.with_name(f"{base}.ver.csv")

def _dep_csv_path(p: Path) -> Path:
    """return the filename base iwth the .dep.csv ending"""
    base = _base_without_csv(p.name)
    return p.with_name(f"{base}.dep.csv")


def preprocess_csv_file(fullpath, assetid, firmwareid, dbname):
    """Prepares sbom csv data for insertion into the database."""
    # Reads version csv file and place in dataframe

    # create the paths for the .ver and .dep files
    ver_path = _ver_csv_path(fullpath)
    dep_path = _dep_csv_path(fullpath)

    # add check to make sure the file exists before preprocessing 
    if not ver_path.exists():
        logger.warning(f"Missing SBOM info csv for {fullpath.name} at {ver_path}, skipping")
        return False
    
    # given the information from the .ver file, preprocess the info for db insertion
    sbom_info = preprocess_sbom_info(ver_path, firmwareid)
    if sbom_info is None or sbom_info.empty:
        logger.warning(f"SBOM info {ver_path.name} has no data, skipping")
        return False
    
    sbomid = insert_into_FirmwareBomVer(dbname, sbom_info)

    component_info = preprocess_component_info(fullpath, firmwareid, sbomid)
    components_no_running = component_info.drop(["Running"], axis=1, inplace=False)
    success = insert_into_firmwareSummary(dbname, components_no_running)
    running_components = component_info[component_info["Running"] == "TRUE"]
    success = ingest_running_components(dbname, assetid, sbomid, running_components)

    # check if the information was inserted successfully and that the sbom has a dependency sbom file (docker scout sboms do not have dependencies)
    if success and dep_path.exists():
        success = preprocess_dep_file(dep_path, sbomid, firmwareid, dbname)

    return success


def process_csv_files(filenames, dirpath, assetid, firmwareid, dbname, score_path):
    """Processes all csv files in directory for insertion into FirmwareSummary table."""
    sboms_to_exclude = [
        str(Path(sbom).with_suffix(".csv")) for sbom in get_excluded_sboms(score_path)
    ]
    for filename in filenames:
        # only process csv files 
        if not filename.lower().endswith(".csv"):
            logger.debug(f"Skipping non-csv file: {filename}")
            continue
        p = Path(filename)
        if p.stem.endswith(".ver") or p.stem.endswith(".dep"):
            continue
        
        full_path = Path(os.path.join(dirpath, filename))
        if str(Path(*full_path.parts[3:])) not in sboms_to_exclude:
            success = preprocess_csv_file(full_path, assetid, firmwareid, dbname)
            if not success:
                logger.warning(f"File: {full_path.name} could not be processed.")


def ingest_sboms(dbname, csv_folder, score_path):
    """Processes all SBOMs in a given csv folder."""
    clear_table(dbname, "FirmwareID")
    assets = get_full_table(dbname, "AssetData")  # Get asset list from database
    for _, asset in assets.iterrows():
        if (
            (asset["Brand"] is None)
            or (asset["Model"] is None)
            or (asset["FirmwareVersion"] is None)
        ):
            continue  # skip file if brand, model or firmwareversion does not exist

        brand = Path(str(asset["Brand"]).strip())
        model = Path(str(asset["Model"]).strip())
        version = Path(str(asset["FirmwareVersion"]).strip())

        # Expected path: /data/csvs/<Brand>/<Model>/<FirmwareVersion>
        firmware_folder = csv_folder / brand / model / version
        logger.info(f"Checking existence of path with SBOMs: {firmware_folder}")

        # Fallback when extra directory exists: /Brand/Model/<FirmwareName>/<FirmwareVersion>
        if not firmware_folder.exists():
            alt_base = csv_folder / brand / model
            try:
                candidates = [
                    d / version for d in alt_base.iterdir()
                    if d.is_dir() and (d / version).exists()
                ]
            except FileNotFoundError:
                candidates = []

            if candidates:
                firmware_folder = candidates[0]
                logger.info(f"Using nested folder containing SBOMs {firmware_folder}")
            else:
                logger.warning(f"No matching folder containing SBOMs was found for {brand}/{model}/{version}")
                continue

        logger.info(f"Ingesting SBOMs in {firmware_folder}")
        filenames = [
            f for f in os.listdir(firmware_folder)
        ]  # Get all firmware files for specified version
        pattern = r"^(.*)\.(NetRise\.cdx|docker-scout\.cdx)\..*\.csv$"
        filenames.sort(key=len)
        match = re.match(pattern, filenames[0], re.IGNORECASE)
        if match:
            firmware_name = str(match.group(1))
        else:
            firmware_name = str(filenames[0])
        df = pd.DataFrame(
            [
                {
                    "Brand": asset["Brand"],
                    "Model": asset["Model"],
                    "FirmwareName": firmware_name,
                    "FirmwareVersion": asset["FirmwareVersion"],
                }
            ]
        )
        firmwareid = insert_into_FirmwareID(dbname, df)
        update_asset_entry(asset["AssetID"], firmwareid, dbname)
        process_csv_files(
            filenames,
            firmware_folder,
            asset["AssetID"],
            firmwareid,
            dbname,
            score_path,
        )


def get_latest_org_id(dbname):
    """Returns the most recently created Organization ID from the Organization table.
    If no organization exists, returns 1 as the default value."""
    try:
        with sqlite3.connect(dbname) as conn:
            cursor = conn.cursor()
            sql = "SELECT MAX(OrgId) FROM Organization"
            cursor.execute(sql)
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
            else:
                return 1
    except Exception as e:
        logger.error(
            f"Database Error - failed to get the latest OrgId from Organization table: {e}"
        )
        return 1


def ingest_assets(dbname, asset_excel_data):
    """Ingests all asset data within a given excel sheet."""
    success = False
    asset_data_columns = [
        "OrgId",
        "AssetID",
        "Name",
        "Names",
        "MAC",
        "MACNorm",
        "Category",
        "Model",
        "Type",
        "Risk",
        "Brand",
        "Users",
        "Sensor",
        "IPv4Address",
        "IPv4AddressNorm",
        "OS",
        "OSVersion",
        "Location",
        "Boundaries",
        "IPv6Address",
        "IPv6AddressNorm",
        "Site",
        "Suspended",
        "Alerts",
        "SerialNum",
        "DeviceID",
        "DataSources",
        "FirmwareName",
        "FirmwareVersion",
        "FirmwareID",
        "Roles",
        "AccessSwitch",
        "PurdueLevel",
        "BusinessImpact",
        "FirstSeen",
        "LastSeen",
        "CollectionToolName",
        "CollectionToolVer",
        "CollectionDateTime",
        "CollectionToolType",
    ]
    asset_inventory = get_sheet_from_excel(asset_excel_data, "AssetInventory")
    if not asset_inventory.empty:
        # Drop columns that are not in AssetData
        asset_inventory = asset_inventory[
            asset_inventory.columns.intersection(asset_data_columns)
        ]
        # OrgId should be the most recently created Organization
        asset_inventory["OrgId"] = get_latest_org_id(dbname)

        if "MAC" in asset_inventory.columns:
            asset_inventory["MACNorm"] = (
                asset_inventory["MAC"]
                .fillna("")
                .astype(str)
                .str.replace(r"[^0-9A-Fa-f]", "", regex=True)
                .str.upper()
            )
        if "IPv4Address" in asset_inventory.columns:
            asset_inventory["IPv4AddressNorm"] = (
                asset_inventory["IPv4Address"]
                .fillna("")
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
            )
        if "IPv6Address" in asset_inventory.columns:
            asset_inventory["IPv6AddressNorm"] = (
                asset_inventory["IPv6Address"]
                .fillna("")
                .astype(str)
                .str.replace(r"[^0-9A-Fa-f]", "", regex=True)
                .str.upper()
            )

        success = insert_into_assetdata(dbname, asset_inventory)
    return success
