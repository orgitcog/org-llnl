# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import sqlite3
from loguru import logger
from typing import Dict, List, Any, Hashable
from pathlib import Path


def table_to_json(dbname: Path, tablename: str) -> List[Dict[Hashable, Any]]:
    df = pd.DataFrame()
    if not tablename in [
        "FirmwareSummaryExport",
        "AssetDataExport",
        "ProgramsExport",
        "OrgExport",
        "SubComponentExport",
    ]:
        logger.error(f"Input Error - requested tablename not valid - {tablename}")
        return []
    with sqlite3.connect(dbname) as conn:
        try:
            sql = f"SELECT * FROM {tablename}"
            df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"Database Error - failed to get table data - {e}")
            return []

    return df.to_dict(orient="records")


def json_to_csv(json_data: List[Dict[Hashable, Any]], csv_path: Path) -> bool:
    try:
        df = pd.DataFrame(json_data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Successfully wrote CSV to {csv_path}")
        return True
    except Exception as e:
        logger.error(f"Pandas Error - could not convert json data to dataframe - {e}")
        return False
