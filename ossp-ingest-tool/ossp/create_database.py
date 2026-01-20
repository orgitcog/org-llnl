# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import os
import requests
import pandas as pd
from loguru import logger
from pathlib import Path

from ossp.database import (
    create_database_from_schema,
    run_sql_script,
    insert_into_organizations,
    insert_into_tmp,
    from_tmp_to_spdxlicense,
    populate_componentlicense_table,
    populate_componentlicense_table_from_alias,
)
from ossp.ingest_csvs import ingest_sboms, ingest_assets


def setup_spdxlicenses_table(dbname: Path):
    """Queries the spdx website and populates spdxlicense table with given data."""
    success = False
    response = requests.get("https://spdx.org/licenses/licenses.json", verify=False)
    spdxlicense_df = pd.DataFrame(response.json()["licenses"])
    spdxlicense_df = spdxlicense_df.astype(
        {
            "reference": "string",
            "isDeprecatedLicenseId": "string",
            "detailsUrl": "string",
            "referenceNumber": "string",
            "name": "string",
            "licenseId": "string",
            "seeAlso": "string",
            "isOsiApproved": "string",
            "isFsfLibre": "string",
        }
    )
    success = insert_into_tmp(dbname, spdxlicense_df)
    if success:
        success = from_tmp_to_spdxlicense(dbname)
    return success


def setup_spdxlicensealias_table(dbname: Path):
    """Populates spdxlicensealias table with a provided sql script."""
    run_sql_script(dbname, Path("ossp/scripts/aliases.sql"))


def refresh_license_info(dbname: Path):
    if setup_spdxlicenses_table(dbname):
        setup_spdxlicensealias_table(dbname)
    success = populate_componentlicense_table(dbname)
    if not success:
        raise Exception("Failed to populate ComponentLicense table.")
    success = populate_componentlicense_table_from_alias(dbname)
    if not success:
        raise Exception("Failed to populate ComponentLicense table from alias.")


def add_organizations(dbname: Path, orgs: list[dict]):
    """Inserts organizations provided in a list of dictionaries."""
    orgs_df = pd.DataFrame(orgs).reset_index(drop=True)
    insert_into_organizations(dbname, orgs_df)


def create_blank_database(dbname: Path, schema: Path):
    """Creates a blank database based on given schema."""
    create_database_from_schema(dbname, schema)
    if not setup_spdxlicenses_table(dbname):
        logger.warning("Failed to setup spdxlicense table.")
    setup_spdxlicensealias_table(dbname)


def create_database(
    dbname: Path,
    schema: Path,
    asset_excel_data: Path,
    csv_folder: Path,
    orgs: list[dict],
    score_path: Path,
):
    """Creates a database according to the provided schema and populates it with user provided data."""
    if os.path.exists(dbname):
        os.remove(dbname)

    create_blank_database(dbname, schema)

    add_organizations(dbname, orgs)

    ingest_assets(dbname, asset_excel_data)

    ingest_sboms(dbname, csv_folder, score_path)

    populate_componentlicense_table(dbname)
    populate_componentlicense_table_from_alias(dbname)

    ## create the views for export
    run_sql_script(dbname, Path("ossp/scripts/views/FirmwareSummaryExport.sql"))
    run_sql_script(dbname, Path("ossp/scripts/views/AssetDataExport.sql"))
    run_sql_script(dbname, Path("ossp/scripts/views/ProgramsExport.sql"))
    run_sql_script(dbname, Path("ossp/scripts/views/OrgExport.sql"))
    run_sql_script(dbname, Path("ossp/scripts/views/SubComponentExport.sql"))

    logger.info("Database Creation Complete.")
