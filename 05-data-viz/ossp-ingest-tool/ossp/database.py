# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import sqlite3
from typing import Optional, Tuple
from loguru import logger
from pathlib import Path
import pandas as pd


def create_database_from_schema(dbname, schema):
    """Create database from a provided sql script."""
    with sqlite3.connect(dbname) as conn:
        with open(schema, "r") as f:
            conn.executescript(f.read())
            conn.commit()


def run_sql_script(dbname: Path, script: Path | str):
    """Run a generic sql script on a given database."""
    with sqlite3.connect(dbname) as conn:
        if isinstance(script, Path):
            with open(script, "r") as f:
                conn.executescript(f.read())
                conn.commit()
        elif isinstance(script, str):
            conn.executescript(script)
            conn.commit()


def clear_table(dbname, tablename):
    """Executes SQL query on database to clear data from a table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = f"DELETE FROM {tablename}"
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error - failed to clear table {tablename} - {e}")


def delete_table(dbname, tablename):
    """Executes SQL query on database to delete table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = f"DROP TABLE IF EXISTS {tablename};"
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error - failed to delete table {tablename} - {e}")


def insert_into_assetdata(dbname, assets):
    """Inserts a given org dataframe into the AssetData table."""
    success = False
    with sqlite3.connect(dbname) as conn:
        try:
            results = assets.to_sql(
                "AssetData",
                conn,
                index=False,
                if_exists="append",
                chunksize=1000,
                method="multi",
            )
            if results > 0:
                success = True
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to insert assets into AssetsData table - {e}"
            )
    return success


def insert_into_organizations(dbname, orgs):
    """Inserts a given org dataframe into the organizations table."""
    success = False
    with sqlite3.connect(dbname) as conn:
        try:
            results = orgs.to_sql(
                "Organization",
                conn,
                index=False,
                if_exists="append",
                chunksize=1000,
                method="multi",
            )
            if results > 0:
                success = True
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to insert organizations into Organization table - {e}"
            )
    return success


def insert_into_firmwareSummary(dbname, df):
    success = False
    with sqlite3.connect(dbname) as conn:
        try:
            results = df.to_sql(
                "FirmwareSummary",
                conn,
                index=False,
                if_exists="append",
                chunksize=1000,
                method="multi",
            )
            if results > 0:
                success = True
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to insert df into FirmwareSummary table - {e}"
            )
    return success


def insert_into_FirmwareID(dbname, firmware_info):
    """Inserts a given firmware_info dataframe into the FirmwareID table."""
    with sqlite3.connect(dbname) as conn:
        try:
            results = firmware_info.to_sql(
                "FirmwareID",
                conn,
                index=False,
                if_exists="append",
                chunksize=1000,
                method="multi",
            )
            if results > 0:
                success = True
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to insert firmware_info into FirmwareID table - {e}"
            )
            return None

    return select_latest_firmwareid(dbname)


def insert_into_FirmwareBomVer(dbname, sbom_info):
    """Inserts a given sbom_info dataframe into the FirmwareBomVer table."""
    with sqlite3.connect(dbname) as conn:
        try:
            results = sbom_info.to_sql(
                "FirmwareBomVer",
                conn,
                index=False,
                if_exists="append",
                chunksize=1000,
                method="multi",
            )
            if results > 0:
                success = True
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to insert sbom_info into FirmwareBomVer table - {e}"
            )
            return None

    return select_latest_sbomid(dbname)


def insert_into_SubComponents(dbname, ParentID, ChildID, rootnode, sbomid):
    """Executes SQL query on database to insert data into the SubComponents Table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "INSERT INTO SubComponent (ParentComponentID, ChildComponentID, RootNode, SBOMID) VALUES (?, ?, ?, ?)"
            cursor.execute(sql, (ParentID, ChildID, rootnode, sbomid))
        except Exception as e:
            logger.error(f"Database Error - failed to insert into SubComponent - {e}")
            return False
    return True


def select_component_id(dbname, sbomid, component_name, license, version):
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "SELECT ComponentID FROM FirmwareSummary WHERE SBOMID=? AND Component=? AND License=? AND Version=? LIMIT 1;"
            params = (
                sbomid,
                component_name,
                license,
                version,
            )
            cursor.execute(sql, params)
            id = cursor.fetchone()
        except Exception as e:
            logger.error(f"Database Error - failed to get latest FirmwareID - {e}")
            return None
    return id if id == None else id[0]


def insert_into_Programs(dbname, AssetID, ProgramName, ProgramComponentID, Running):
    """Executes SQL query on database to insert data into the SubComponents Table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "INSERT INTO Programs (AssetID, ProgramName, ProgramComponentID, Running) VALUES (?, ?, ?, ?)"
            cursor.execute(sql, (AssetID, ProgramName, ProgramComponentID, Running))
        except Exception as e:
            logger.error(f"Database Error - failed to insert into Programs - {e}")
            return False
    return True


def get_deps_from_FirmwareSummary(dbname, bomrefs, firmwareid, sbomid):
    """Executes SQL query on database to get dependecy info from the FirmwareSummary Table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = f"""SELECT BomRef, ComponentID FROM FirmwareSummary WHERE BomRef IN ({', '.join(['?' for _ in bomrefs])}) AND FirmwareID = ? AND SBOMID = ?; """
            cursor.execute(sql, (*bomrefs, firmwareid, sbomid))
            values = cursor.fetchall()
            depdict = dict(values)
        except Exception as e:
            logger.error(
                f"Database Error - failed to get BomRef, ComponentId from FirmwareSummary table - {e}"
            )
            return None
    return depdict


def from_tmp_to_assetdata(dbname):
    """Insert data into AssetData by copying selected columns from tmp and supplying constant values for the rest."""
    with sqlite3.connect(dbname) as conn:
        try:

            logger.debug("Inserting data into AssetData from tmp table")

            cursor = conn.cursor()
            sql = """
            INSERT INTO AssetData (OrgID, Name, MAC, Model, Type, Brand, IPv4Address, 
                                   Location, Site, FirmwareVersion, PurdueLevel, FirstSeen, 
                                   LastSeen, CollectionToolName, CollectionToolVer, CollectionDateTime)
            SELECT ?, Name, MAC, Model, Type, Brand, IPv4Address, Location, Site, FirmwareVersion, 
                   ?, ?, ?, ?, ?, ?
            FROM tmp
            """
            params = (
                1,  # Constant placeholder for OrgID
                1,  # Constant placeholder for PurdueLevel (example)
                "01/22/2025 22:10:15",  # Constant placeholder for FirstSeen
                "01/22/2025 22:10:15",  # Constant placeholder for LastSeen
                "Manual",  # Constant placeholder for CollectionToolName
                "V1.0",  # Constant placeholder for CollectionToolVer
                "01/22/2025 22:10:15",  # Constant placeholder for CollectionDateTime
            )
            cursor.execute(sql, params)

            sql = """
            UPDATE AssetData SET OrgId=1, PurdueLevel=1, FirstSeen='01/22/2025 22:10:15', LastSeen='01/22/2025 22:10:15', CollectionToolName='Manual', CollectionToolVer='V1.0', CollectionDateTime='01/22/2025 22:10:15'
            """
            cursor.execute(sql, params)

            conn.commit()
        except Exception as e:
            logger.error(f"Database Error - failed to insert into AssetData - {e}")
            return False
    return True


def from_tmp_to_firmwareid(dbname):
    """Executes SQL query on database to move data from tmp table to FirmwareID Table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO FirmwareID (Brand, Model, FirmwareName, FirmwareVersion) 
            SELECT Brand, Model, FirmwareName, FirmwareVersion FROM tmp """
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error - failed to insert into FirmwareID - {e}")
            return False
    return True


def from_tmp_to_firmwaresummary(dbname):
    """Executes SQL query on database to move data from tmp table to FirmwareSummary Table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = """INSERT INTO FirmwareSummary (FirmwareID, Component, ComponentType, License, Version, CPE, PURL, BomRef, ExternalLicenseRef, Author) 
            SELECT FirmwareID, name, type, license, version, cpe, purl, bomref, externalReference, author FROM tmp where (type != "file")"""
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to insert into FirmwareSummary - {e}"
            )
            return False
    return True


def from_tmp_to_spdxlicense(dbname):
    """Executes SQL query on database to move data from tmp table into the spdxlicense table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "INSERT OR IGNORE INTO spdxlicenses SELECT reference,isDeprecatedLicenseId,detailsUrl,referenceNumber,name,licenseId,isOsiApproved,isFsfLibre,seeAlso from tmp"
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error - failed to insert into spdxlicenses - {e}")
            return False
    return True


def insert_into_tmp(dbname, df, if_exists="replace"):
    """Inserts a pandas dataframe into the tmp table"""
    # Insert into tmp Table
    success = False
    with sqlite3.connect(dbname) as conn:
        results = df.to_sql(
            "tmp",
            conn,
            if_exists=if_exists,
            index=False,
            chunksize=1000,
            method="multi",
        )
        if results > 0:
            success = True
        conn.commit()
    return success


def select_latest_firmwareid(dbname):
    """Executes a SQL query to grab the last FirmwareID that was added to the FirmwareID table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "SELECT FirmwareID FROM FirmwareID ORDER BY FirmwareID DESC LIMIT 1;"
            cursor.execute(sql)
            result = cursor.fetchone()
        except Exception as e:
            logger.error(f"Database Error - failed to get latest FirmwareID - {e}")
            return None
    return result if result == None else result[0]


def select_latest_sbomid(dbname):
    """Executes a SQL query to grab the last SBOMID that was added to the FirmwareBomVer table."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "SELECT SBOMID FROM FirmwareBomVer ORDER BY SBOMID DESC LIMIT 1;"
            cursor.execute(sql)
            result = cursor.fetchone()
        except Exception as e:
            logger.error(f"Database Error - failed to get latest SBOMID - {e}")
            return None
    return result if result == None else result[0]


def update_asset_entry(assetid, firmwareid, dbname):
    """Updates AssetData with correct FirmwareID."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = "UPDATE AssetData SET FirmwareID = ? WHERE AssetID = ? AND (FirmwareID IS NULL OR LENGTH(FirmwareID)=0)"
            params = (
                firmwareid,
                assetid,
            )
            cursor.execute(sql, params)
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to update assetdata for {assetid} - {e}"
            )
            return False
    return True


def populate_componentlicense_table(dbname):
    """Executes several queries to populate the ComponentLicense Table"""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = """WITH RECURSIVE split(FirmwareID, ComponentID, license_name, rest) AS ( 
                    SELECT FirmwareID, ComponentID, '', License || ',' FROM FirmwareSummary WHERE ComponentID 
                    UNION ALL 
                    SELECT FirmwareID, ComponentID,  
                            substr(rest, 0, instr(rest, ',')), 
                            substr(rest, instr(rest, ',')+1) 
                        FROM split 
                    WHERE rest <> '')
                    INSERT OR IGNORE INTO ComponentLicense SELECT FirmwareId, ComponentID, license_name, NULL 
                    FROM split  
                    WHERE license_name <> '' 
                    ORDER BY  FirmwareID, ComponentID, license_name 
                """
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(f"Database Error - failed to update ComponentLicense - {e}")
            return False
        try:
            sql = """UPDATE ComponentLicense SET SPDXReferenceNum = c.referenceNumber 
            FROM (SELECT referenceNumber,licenseId FROM spdxlicenses) AS c 
            WHERE UPPER(RTRIM(c.licenseId)) = UPPER(RTRIM(ComponentLicense.License)) COLLATE NOCASE"""
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to correlate ComponentLicense licenses to spdxlicense Ids - {e}"
            )
            return False
        try:
            sql = """UPDATE FirmwareSummary 
            SET Export=1 
            FROM (SELECT SPDXReferenceNum,ComponentID,FirmwareID FROM ComponentLicense WHERE SPDXReferenceNum IS NOT NULL) AS c 
            WHERE FirmwareSummary.FirmwareID = c.FirmwareID and FirmwareSummary.ComponentID = c.ComponentID"""
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to update FirmwareSummary Export column - {e}"
            )
            return False
    return True


def populate_componentlicense_table_from_alias(dbname):
    """Exceute SQL queries to udpate the ComponentLicense Table with alias translations."""
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            sql = """UPDATE ComponentLicense 
            SET SPDXReferenceNum = c.referenceNumber 
            FROM (SELECT referenceNumber,spdxlicenses.licenseId,spdxlicensealias.alias FROM spdxlicenses join spdxlicensealias ON UPPER(ltrim(rtrim(spdxlicensealias.licenseId))) = UPPER(rtrim(ltrim(spdxlicenses.licenseId)))) AS c 
            WHERE UPPER(ltrim(rtrim(c.alias))) = UPPER(ltrim(rtrim(ComponentLicense.License))) COLLATE NOCASE
            """
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to update ComponentLicense with spdxlicensealias- {e}"
            )
            return False
        try:
            sql = """UPDATE FirmwareSummary 
            SET Export=1 
            FROM (SELECT SPDXReferenceNum,ComponentID,FirmwareID FROM ComponentLicense WHERE SPDXReferenceNum IS NOT NULL) AS c 
            WHERE FirmwareSummary.FirmwareID = c.FirmwareID and FirmwareSummary.ComponentID = c.ComponentID """
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            logger.error(
                f"Database Error - failed to update FirmwareSummary Export column - {e}"
            )
            return False
    return True


def get_full_table(dbname: Path, tablename: str) -> pd.DataFrame:
    """Exceute SQL query to get asset list."""
    df = pd.DataFrame()
    if not tablename in [
        "AssetData",
        "ComponentLicense",
        "FirmwareID",
        "FirmwareSummary",
        "Organization",
        "Programs",
        "spdxlicensealias",
        "spdxlicenses",
        "SubComponent",
        "FirmwareBomVer",
    ]:
        return df
    with sqlite3.connect(dbname) as conn:
        try:
            sql = f"SELECT * FROM {tablename}"
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(
                f"Database Error - failed to get asset list from database - {e}"
            )
    return df


def get_asset_list(dbname):
    """Exceute SQL query to get asset list."""
    df = pd.DataFrame()
    with sqlite3.connect(dbname) as conn:
        try:
            sql = """SELECT Name FROM AssetData"""
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(
                f"Database Error - failed to get asset list from database - {e}"
            )
    return df


def get_manufacturer_list(dbname):
    """Exceute SQL query to get manufacturer list."""
    df = pd.DataFrame()
    with sqlite3.connect(dbname) as conn:
        try:
            sql = """SELECT Brand FROM AssetData"""
            df = pd.read_sql_query(sql, conn)
            df = df.drop_duplicates()
        except Exception as e:
            logger.error(
                f"Database Error - failed to get asset list from database - {e}"
            )
    return df


##----------------------Research Question Queries-----------------------------##
def rq1_query(dbname: Path, firmwareid: Optional[int] = None) -> pd.DataFrame:
    """Executes query for research question 1 and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = """
                    SELECT
                    ad.AssetID,
                    ad.Name                        AS AssetName,
                    COUNT(DISTINCT fs.ComponentID) AS UniqueExportedComponents
                    FROM
                    AssetData       AS ad
                    LEFT JOIN FirmwareSummary AS fs
                        ON fs.FirmwareID = ad.FirmwareID
                    AND COALESCE(fs.Export, 0) = 1         -- only “exported” components
                    GROUP BY
                    ad.AssetID,
                    ad.Name
                    ORDER BY
                    UniqueExportedComponents DESC,
                    ad.AssetID;
                """
            if firmwareid is not None:
                sql += "WHERE tc.FirmwareID = ? "
                params = (firmwareid,)

            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()  # Return empty df if database error occurs
    return df


def rq1a_query(dbname: Path, firmwareid: Optional[int] = None) -> pd.DataFrame:
    """Executes query for research question 1a and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        params: tuple = ()
        try:
            sql = f"""
            	SELECT 
                    FirmwareID,
                    Component,
                    GROUP_CONCAT(DISTINCT Version) AS Versions, 
                    COUNT(DISTINCT Version) as VersionCount
                FROM 
                    FirmwareSummary """
            if firmwareid is not None:
                sql += "WHERE FirmwareID=? AND Version IS NOT NULL AND Version != '' "
                params = (firmwareid,)
            else:
                sql += "WHERE Version IS NOT NULL AND Version != '' "
                params = ()
            sql += """
                AND Export = 1
                GROUP BY 
                    FirmwareID, Component
                HAVING 
                    COUNT(DISTINCT Version) > 1
                ORDER BY 
                    FirmwareID, VersionCount DESC;"""
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq1b_query(dbname: Path, firmwareid: Optional[int] = None) -> pd.DataFrame:
    """Executes query for research question 1b and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""SELECT fid.Brand, fid.Model, Component, ComponentID, Version, p.Running
            FROM FirmwareSummary fs, FirmwareID fid
            JOIN Programs p ON fs.ComponentID = p.ProgramComponentID
            WHERE p.Running = '1' AND fs.FirmwareID = fid.FirmwareID AND fs.Export = 1"""
            if firmwareid is not None:
                sql += "AND FirmwareID = ?"
                params = (firmwareid,)
            sql += """
            GROUP BY 
                fs.FirmwareID, Component"""
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq1c_query(
    dbname: Path, firmwareid: Optional[int] = None, component: Optional[str] = None
) -> pd.DataFrame:
    """Executes query for research question 1c and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""SELECT ComponentID, Component, Version, License, Author, CPE, PURL 
            FROM FirmwareSummary """
            if firmwareid is not None:
                sql += " WHERE FirmwareID = ?"
                params = params + (firmwareid,)
            if component is not None and firmwareid is not None:
                sql += "AND Component = ?"
                params = params + (component,)
            elif component is not None and firmwareid is None:
                sql += "WHERE Component = ?"
                params = params + (component,)
            sql += "AND Export = 1 GROUP BY Author"
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq2_query(
    dbname: Path, firmwareid: Optional[int] = None, search: Optional[str] = None
):
    """Executes query for research question 2 and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""SELECT DISTINCT AssetID, Name, CPE, PURL, Component, Version, Author FROM AssetData JOIN FirmwareSummary ON AssetData.FirmwareID = FirmwareSummary.FirmwareID"""
            if firmwareid is not None:
                sql += " WHERE FirmwareSummary.FirmwareID = ?"
                params = params + (firmwareid,)
            if search is not None and firmwareid is not None:
                sql += " AND FirmwareSummary.Component LIKE ?;" ""
                params = params + (search,)
            elif search is not None and firmwareid is None:
                sql += (
                    " WHERE FirmwareSummary.Export = 1 AND FirmwareSummary.Component LIKE ?;"
                    ""
                )
                params = params + (search,)
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3_query(dbname): ...


def rq3a_query(dbname: Path) -> pd.DataFrame:
    """Executes query for research question 3a and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            sql = f"""
            SELECT Component, COUNT(*) as ComponentCount
            FROM FirmwareSummary
            WHERE Export = 1
            GROUP BY Component
            ORDER BY ComponentCount DESC;"""
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3b1_query(dbname: Path, component: Optional[str] = None) -> pd.DataFrame:
    """Executes query for research question 3b and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""
            SELECT DISTINCT AssetID, Name, Component, Version, Author
            FROM AssetData
            JOIN FirmwareSummary ON AssetData.FirmwareID = FirmwareSummary.FirmwareID """
            if component is not None:
                sql += (
                    "WHERE FirmwareSummary.Component = ? AND FirmwareSummary.Export = 1"
                )
                params = params + (component,)
            else:
                sql += "WHERE FirmwareSummary.Export = 1"
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3b2_query(dbname: Path, component: Optional[str] = None) -> pd.DataFrame:
    """Executes query for research question 3b and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""
            SELECT Organization.CISector, COUNT(*) as instancesCount
            FROM AssetData
            JOIN FirmwareSummary ON AssetData.FirmwareID = FirmwareSummary.FirmwareID
            JOIN Organization ON AssetData.OrgId = Organization.OrgId """
            if component is not None:
                sql += "WHERE FirmwareSummary.Component = ? AND FirmwareSummary.Export = 1 "
                params = (component,)
            else:
                sql += "WHERE FirmwareSummary.Export = 1 "
            sql += "GROUP BY Organization.CISector"
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3b3_query(dbname: Path, component: Optional[str] = None) -> pd.DataFrame:
    """Executes query for research question 3b and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""
            SELECT AssetData.type, COUNT(*) as zlibInstancesCount
            FROM AssetData 
            JOIN FirmwareSummary ON AssetData.FirmwareID = FirmwareSummary.FirmwareID """
            if component is not None:
                sql += "WHERE FirmwareSummary.Component = ? AND FirmwareSummary.Export = 1 "
                params = (component,)
            else:
                sql += "WHERE FirmwareSummary.Export = 1 "
            sql += "GROUP BY AssetData.type"
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3b4_query(dbname: Path, component: Optional[str] = None) -> pd.DataFrame:
    """Executes query for research question 3b and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            sql = f"""
            SELECT FirmwareID.FirmwareName, FirmwareID.FirmwareVersion, COUNT(*) as zlibInstancesCount
            FROM FirmwareID, FirmwareSummary """
            if component is not None:
                sql += "WHERE FirmwareSummary.Component = ? "
                params = (component,)
            sql += """AND FirmwareSummary.FirmwareID = FirmwareID.FirmwareID AND FirmwareSummary.Export = 1
            GROUP BY FirmwareID.FirmwareName, FirmwareID.FirmwareVersion"""
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3b5_query(dbname: Path, component: Optional[str] = None) -> pd.DataFrame:
    """Executes query for research question 3b and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            sql = f"""
            SELECT FirmwareID.FirmwareName, FirmwareID.FirmwareVersion, FirmwareSummary.ComponentID, FirmwareSummary.Component, FirmwareSummary.version
            FROM FirmwareID, FirmwareSummary WHERE
            """
            if component is not None:
                sql += "FirmwareSummary.Component = ? AND "
                params = (component,)
            sql += """FirmwareSummary.FirmwareID = FirmwareID.FirmwareID AND FirmwareSummary.Export = 1"""
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3c_query(dbname: Path) -> pd.DataFrame:
    """Executes query for research question 3c and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            sql = f"""
            SELECT  ProgrammingLang, COUNT(*)
            FROM FirmwareSummary
            GROUP BY ProgrammingLang"""
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq3d_query(dbname, author: Optional[str] = None) -> pd.DataFrame:
    """Executes query for research question 3d and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""
            SELECT DISTINCT Component, Version, Author, Publisher
            FROM FirmwareSummary """
            if author is not None:
                sql += "WHERE Author LIKE ?"
                params = params + (author,)
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def rq4_query(dbname, sbomid: str) -> pd.DataFrame:
    """Executes query for research question 4 and returns a dataframe with results"""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = f"""
            SELECT ParentComponentID, ChildComponentID, RootNode, SBOMID 
            FROM SubComponent 
            WHERE SBOMID = ?"""
            params = (sbomid,)
            df = pd.read_sql_query(sql, conn, params=params)
        except Exception as e:
            logger.error(f"Database Error - failed to execute query - {e}")
            df = pd.DataFrame()
    return df


def select_firmwareid(dbname, sbomid) -> Optional[int]:
    """Executes a SQL query to grab a firmwareid based on an sbomid."""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = "SELECT FirmwareID FROM FirmwareSummary WHERE SBOMID = ?"
            params = (sbomid,)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchone()
            if result is not None:
                firmware_id = int(result[0])
            else:
                firmware_id = None
        except Exception as e:
            logger.error(f"Database Error - failed to get firmwareid - {e}")
            firmware_id = None
    return firmware_id


def select_FirmwareName(dbname, firmwareid) -> Optional[str]:
    """Executes a SQL query to grab the firmwarename based on a firmwareid."""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = "SELECT Brand, Model FROM FirmwareID where FirmwareID = ?"
            params = (firmwareid,)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchone()
            if result is not None:
                brand, model = result
                return f"{brand} {model}"
            else:
                return None
        except Exception as e:
            logger.error(f"Database Error - failed to get Brand, Model - {e}")
            return None


def select_ComponentInfo(
    dbname, componentid
) -> Optional[Tuple[str, str, str, str, str, str]]:
    """Executes a SQL query to grab the firmwarename based on a firmwareid."""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = "SELECT Component, BomRef, ComponentType, Version, Author, ProgrammingLang FROM FirmwareSummary where ComponentID = ?"
            params = (componentid,)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchone()
            if result is not None:
                return result
            else:
                logger.info(f"No component found for ComponentID: {componentid}")
                return None
        except Exception as e:
            logger.error(f"Database Error - failed to get Component Name - {e}")
            return None


def select_filename_from_sbomid(dbname, sbomid) -> Optional[str]:
    """Executes a SQL query to grab the filename based on an sbomid."""
    with sqlite3.connect(dbname) as conn:
        try:
            params: tuple = ()
            sql = "SELECT FileName FROM FirmwareBomVer where SBOMID = ?"
            params = (sbomid,)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            result = cursor.fetchone()
            if result is not None:
                return str(result[0])
            else:
                logger.info(f"No filename found for sbomid: {sbomid}")
                return None
        except Exception as e:
            logger.error(f"Database Error - failed to get filename - {e}")
            return None


def infer_and_update_export(dbname):
    """
    For each row in FirmwareSummary with Export=0, update Export to 1 if there are
    other rows with the same Componentthat have Export=1.
    """
    logger.info("Inferring and updating Export values in FirmwareSummary table.")
    with sqlite3.connect(dbname) as conn:
        try:
            cursor = conn.cursor()
            # Find all unique (Component, Version, License) where Export=1
            cursor.execute(
                """
                SELECT DISTINCT Component, Version, License
                FROM FirmwareSummary
                WHERE Export = 1
            """
            )
            export_ones = set(cursor.fetchall())

            # For each (Component, Version, License) with Export=1, update Export=0 rows to 1
            for comp, ver, lic in export_ones:
                cursor.execute(
                    """
                    UPDATE FirmwareSummary
                    SET Export = 1
                    WHERE Export = 0
                      AND Component = ?               
                """,
                    (comp,),
                )
            conn.commit()
            logger.info("Export values inferred and updated successfully.")

        except Exception as e:
            logger.error(f"Database Error - failed to infer/update Export values - {e}")
            return False
    return True

def get_view_columns(db_path, view_name):
    """
    For each view name in the database, return the column names.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({view_name})")
        columns = [row[1] for row in cursor.fetchall()]
    return columns

def get_view_values(db_path, view_name):
    """
    For each view name, return the unique values in each row. 
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {view_name}")
        rows = cursor.fetchall()
        unique_values = list({item for row in rows for item in row})
    return unique_values