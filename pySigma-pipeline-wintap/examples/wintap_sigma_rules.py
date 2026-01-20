import argparse
import heapq
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tqdm
import wintappy.datautils.rawutil as ru
from duckdb import BinderException, CatalogException
from jinja2 import BaseLoader, Environment
from jinja2.exceptions import TemplateSyntaxError
from sigma.backends.duckdb import DuckDBBackend
from sigma.collection import SigmaCollection, SigmaRule
from sigma.processing.resolver import ProcessingPipelineResolver
from slugify import slugify
from wintappy.database.wintap_duckdb import WintapDuckDB, WintapDuckDBOptions
from wintappy.etlutils.utils import configure_basic_logging, daterange, get_date_range

from sigma.pipelines.wintap.wintap import wintap_pipeline

# Constants
SIGMA_RULES_PATH = "sigma_rules"
VIEW_CREATION_QUERIES = [
    """
    create or replace table joined_process as
    select p.*, pp.args as parent_command_line, pp.process_name as parent_process_name,
    [p.file_md5, p.file_sha2] as hashes
    from process as p
    join process as pp on p.parent_pid_hash=pp.pid_hash and p.daypk=pp.daypk
    """,
    """
    create or replace table sigma_process_image_load as
    select p.process_path, p.args, ri.md5 hashes, i.*
    from process_image_load i
    join raw_imageload ri on i.pid_hash=ri.pidhash and i.filename=ri.filename and i.daypk=ri.daypk
    join process p on i.pid_hash=p.pid_hash and i.daypk=p.daypk
    """,
    """
    create or replace table sigma_process_registry as
    select p.process_path, r.*
    from process_registry as r
    join process as p on r.pid_hash=p.pid_hash and r.daypk=p.daypk
    """,
    """
    create or replace table sigma_process_net_conn as
    select p.process_path, pnc.*
    from process_net_conn as pnc
    join process as p on pnc.pid_hash=p.pid_hash and pnc.daypk=p.daypk
    """,
]


def format_rules(
    rules: SigmaCollection, duckdb_backend: DuckDBBackend
) -> Tuple[Dict[str, SigmaRule], Dict[str, Any]]:
    analytics = {}
    queries = {}
    for rule in rules.rules:
        rule_id = rule.id if rule.id else slugify(rule.title)
        analytics[rule_id] = rule
        queries[rule_id] = duckdb_backend.convert(SigmaCollection(rules=[rule]), "wintap")
    return analytics, queries


def wintap_setup(data_set_path: str) -> WintapDuckDB:
    con = ru.init_db(data_set_path)
    options = WintapDuckDBOptions(con, data_set_path, False)
    db = WintapDuckDB(options)
    for view in VIEW_CREATION_QUERIES:
        logging.info(f"  Executing: {view}")
        db.query(view)
    return db


def process_range(
    wintap_db: WintapDuckDB,
    queries: Dict[str, Any],
    env: Environment,
    start_date: datetime,
    end_date: datetime,
    data_set_path: str,
    store_data: bool = False,
) -> None:
    for single_date in tqdm.tqdm(daterange(start_date, end_date), desc="Processing DayPK"):
        daypk = int(single_date.strftime("%Y%m%d"))
        for analytic_id, query in tqdm.tqdm(queries.items(), desc="Exec Queries", leave=False):
            # This is a workaround for now, as this will mess up jinja templating
            if not query or "{%" in query or "{ %" in query:
                continue
            try:
                query_str = env.from_string(query).render({"search_day_pk": daypk})
            except TemplateSyntaxError:
                logging.warning(query)
                continue
            if store_data:
                with open(f"{SIGMA_RULES_PATH}/{analytic_id}.sql", "w") as f:
                    f.write(query_str)
            try:
                wintap_db.query(
                    f"INSERT INTO sigma_labels SELECT pid_hash, '{analytic_id}', first_seen, 'pid_hash' FROM ({query_str})"
                )
            except (CatalogException, BinderException) as err:
                logging.error(f"{analytic_id}: {err.args}", stack_info=False)
        wintap_db.write_table("sigma_labels", daypk, location=data_set_path)
        wintap_db.clear_table("sigma_labels")


def setup_logging(log_level: str):
    configure_basic_logging()
    try:
        logging.getLogger().setLevel(log_level)
    except ValueError:
        logging.error(f"Invalid log level: {log_level}")
        sys.exit(1)


def pk_sort(pk):
    if "=" in pk:
        _, value = pk.split("=")
        return value
    return pk


def date_range(data_set_path: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Return start and end datetimes for the given path. This path must be a date partitioned agg_level: raw|rolling.
    If there is no data at all, returns None.
    """
    DEFAULT_DATE_RANGE_PATH = f"raw_sensor{os.sep}raw_process{os.sep}"
    path = f"{data_set_path}{os.sep}{DEFAULT_DATE_RANGE_PATH}"
    try:
        daypks = os.listdir(path)
    except FileNotFoundError:
        # directory does not exist
        logging.info(f"Directory ({path}) does not exist. Will use default times.")
        daypks = []
    # remove "bad" directories
    daypks = [d for d in daypks if "=" in d]
    # if there is no data, return a default of a day ago
    if len(daypks) == 0:
        print(f"No daypks in {data_set_path}{os.sep}{DEFAULT_DATE_RANGE_PATH}")
        return None, None

    _, start_day = heapq.nsmallest(1, daypks, key=pk_sort)[0].split("=")
    _, end_day = heapq.nlargest(1, daypks, key=pk_sort)[0].split("=")
    return datetime.strptime(f"{start_day}", "%Y%m%d"), datetime.strptime(
        f"{int(end_day)+1}", "%Y%m%d"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="wintap_sigma_rules.py",
        description="Run sigma enrichments against wintap data, write out lookups and results. Example usage: `python wintap_sigma_rules.py /path/to/sigma/rules /path/to/dataset -l DEBUG -s`",
    )
    parser.add_argument(
        "-r",
        "--sigma-input-rules-path",
        help="Path to the directory containing Sigma rules",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--data-set-path",
        help="Path to the dataset",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--store-sql",
        action="store_true",
        help="store generated sql in a non-tracked git dir, sigma_rules/",
    )
    parser.add_argument(
        "-l", "--log-level", help="Logging Level: INFO, WARN, ERROR, DEBUG", default="INFO"
    )
    args, _ = parser.parse_known_args(argv)
    setup_logging(args.log_level)
    if args.store_sql:
        Path(SIGMA_RULES_PATH).mkdir(exist_ok=True)

    # Create the pipeline resolver
    logging.info("Configuring pipeline")
    piperesolver = ProcessingPipelineResolver()
    piperesolver.add_pipeline_class(wintap_pipeline())
    combined_pipeline = piperesolver.resolve(piperesolver.pipelines)
    duckdb_backend = DuckDBBackend(combined_pipeline, collect_errors=True)

    # Load Sigma rules
    logging.info("Loading SIGMA rules")
    rules = SigmaCollection.load_ruleset([args.sigma_input_rules_path], collect_errors=True)

    # Setup Wintap database
    logging.info("Opening database")
    wintap_db = wintap_setup(args.data_set_path)

    # Format rules and queries
    logging.info("Formatting rules")
    _, queries = format_rules(rules, duckdb_backend)

    # Define date range for processing
    logging.info("Calculating date range to process")
    start_date, end_date = date_range(args.data_set_path)
    env = Environment(loader=BaseLoader)

    # Process the date range
    process_range(wintap_db, queries, env, start_date, end_date, args.data_set_path, args.store_sql)


if __name__ == "__main__":
    main(argv=None)
