# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from flask import Flask, render_template, send_file, request, redirect, url_for, flash
from pathlib import Path
import shutil
import os
import traceback
import json
import pandas as pd
from shutil import make_archive
import tempfile
import numpy as np
import time
import tomli

from ossp.process_sboms import process_all_sboms, rename_all_sboms
from ossp.create_database import create_database, refresh_license_info
from ossp.research_questions import *
from ossp.database import run_sql_script, infer_and_update_export
from ossp.sbomqs.sbom_scoring import (
    score_sboms,
    get_scores_from_file,
    filter_scores_by_min,
    save_sbom_filter,
)
from ossp.redaction.redact import (
    create_column_spec,
    create_view_values,
    redact,
    build_policy_from_rules,
    policy_to_rules,
    upsert_policy,
    normalize_policy_dict,
)


# app = Flask(__name__)
# ------------------------------------Constants---------------------------------------#
BLANK_SBOM_ZIP = Path("/data/blank.zip")
SBOMS_FOLDER = Path("/data/SBOMs")
CSVS_FOLDER = Path("/data/csvs")
ORG_DATA = Path("/data/user_data.json")
ASSET_DATA = Path("/data/AssetData.xlsx")
DATABASE = Path("/data/databases/database.db")
SCHEMA = Path("/data/databases/schema.sql")
SCORES = Path("/data/databases/scores.json")
SCORES_XLSX = Path("/data/databases/scores.xlsx")
SBOMS_IN_USE = Path("/data/databases/sboms_in_use.json")
REDACT_TOML = Path("ossp/redaction/redact.toml")
REDACTED_DATA = Path("/data/databases/redacted")
PROGRESS = (None, "")


# ------------------------------------Helper Functions---------------------------------------#
# TODO: Implement this function
def validate_directory_structure(directory):
    """Helper function to validate directory structure"""
    return True


def validate_asset_file(file):
    """Helper function to check xlsx magic bytes of uploaded file."""
    if os.path.exists(file):
        with open(file, "rb") as f:
            sig = f.read(4)
        if sig == b"\x50\x4b\x03\x04":
            return True
    return False


def create_zip_file(zip_name: Path, structure: dict):
    """Helper function to create a zip folder based on given structure"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        for brand, subfolders in structure.items():
            brand_folder = tmpdirname / Path(brand)
            brand_folder.mkdir()
            for model, subfolders in subfolders.items():
                model_folder = tmpdirname / brand_folder / Path(model)
                model_folder.mkdir()
                for device in subfolders:
                    device_folder = (
                        tmpdirname / brand_folder / model_folder / Path(device)
                    )
                    device_folder.mkdir()

        make_archive(str(zip_name), "zip", tmpdirname)


def create_structure_from_dataframe(asset_df: pd.DataFrame) -> dict:
    """Helper function to create structure from a dataframe"""
    if not all(
        value in asset_df.columns for value in ["Brand", "Model", "FirmwareVersion"]
    ):
        logger.error(
            "User Input Error: Excel sheet does not contain Brand, Model, and FirmwareVersion Columns"
        )
    structure: dict = {}
    for _, row in asset_df.iterrows():
        brand, model, firmwareversion = (
            str(row["Brand"]),
            str(row["Model"]),
            str(row["FirmwareVersion"]),
        )
        if row["Brand"] is not np.nan:
            if brand not in structure.keys():
                structure[brand] = {}
            if row["Model"] is not np.nan:
                if model not in structure[brand].keys():
                    structure[brand][model] = []
                if row["FirmwareVersion"] is not np.nan:
                    if firmwareversion not in structure[brand][model]:
                        structure[brand][model].append(str(firmwareversion))
    return structure


def generate_csvs():
    """Helper function to generate csvs"""
    logger.info(f"Generating CSVs from {SBOMS_FOLDER}")
    start = time.perf_counter()
    process_all_sboms(SBOMS_FOLDER, CSVS_FOLDER, SCORES)
    logger.info(f"CSVs generated in {(time.perf_counter()-start):.3f} seconds")


def populate_database():
    """Helper function to populate database."""
    start = time.perf_counter()
    with open(ORG_DATA, "r") as json_file:
        orgs = json.load(json_file)
    create_database(DATABASE, SCHEMA, ASSET_DATA, CSVS_FOLDER, [orgs], SCORES)
    logger.info(f"Database populated in {(time.perf_counter()-start):.3f} seconds")


def analysis():
    """Helper function to run analysis."""
    start = time.perf_counter()
    questions = {
        1: [rq1, rq1a, rq1b, rq1c],
        2: [rq2],
        3: [rq3a, rq3b, rq3c, rq3d],
        4: [rq4],
    }

    rq_results = {1: [], 2: [], 3: [], 4: []}
    for question, queries in questions.items():
        for query in queries:
            result = query(DATABASE)
            if isinstance(result, list):
                rq_results[question] += result
            else:
                rq_results[question].append(result)

    with open("results.json", "w") as f:
        json.dump(rq_results, f)
    logger.info(f"Analysis run in {(time.perf_counter()-start):.3f} seconds")


# ------------------------------------Routes---------------------------------------#


def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
        """Returns Home Page"""
        return render_template("index.html")

    @app.route("/results", methods=["GET"])
    def get_results_page():
        """Returns Results Page"""
        return render_template("results.html")

    @app.route("/scores", methods=["GET"])
    def get_scores_page():
        """Returns Scores Page"""
        return render_template("scores.html")

    @app.route("/redact", methods=["GET"])
    def get_redaction_page():
        """Returns Redact page with values to populate dropdowns"""
        columns = create_column_spec()
        values = create_view_values() 

         # Normalize to strings to avoid TypeError when sorting mixed types
        def to_str(x):
            return "" if x is None else str(x)

        # Column names are usually strings, but normalize for safety
        key_names = sorted({
            to_str(col["name"]) for view in columns for col in view.get("columns", [])
        })

        # Field values can be mixed types; normalize then sort
        field_values = sorted({
            to_str(val["name"]) for view in values for val in view.get("values", [])
        })


        return render_template("redact.html", key_names=key_names, field_values=field_values, row_values=field_values, multi_kv_keys=key_names, multi_kv_values=field_values, current_rules=[])
    
    @app.route("/redact/presets", methods=["GET"])
    def list_redaction_presets():
        """Return available policy names from redact.toml"""
        try:
            with open(REDACT_TOML, "rb") as fp:
                config = tomli.load(fp)
            return {"presets": list(config.keys())}, 200
        except FileNotFoundError:
            return {"presets": []}, 200
        except Exception:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500

    @app.route("/redact/rules", methods=["GET"])
    def get_rules_for_preset():
        """Return rules array for a given preset name"""
        name = request.args.get("preset", "default")
        try:
            rules = policy_to_rules(REDACT_TOML, name)
            return {"rules": rules}, 200
        except FileNotFoundError:
            return {"rules": []}, 200
        except KeyError:
            return {"error": f"Preset '{name}' not found"}, 404
        except Exception:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        
    @app.route("/redact/save", methods=["POST"])
    def save_redaction_preset():
        """Save current UI rules under a user provided preset name, preserving [default]."""
        name = (request.form.get("name", "") or "").strip()
        if not name:
            return {"error": "Preset name is required"}, 400
        if name.lower() == "default":
            return {"error": "Saving to 'default' is not allowed"}, 400

        rules_json = request.form.get("rules_json", "[]")
        try:
            rules = json.loads(rules_json)
        except Exception:
            return {"error": "Invalid rules_json"}, 400

        try:
            policy_dict = build_policy_from_rules(rules)
            upsert_policy(REDACT_TOML, name, policy_dict)
            return {"message": "Preset saved", "name": name}, 200
        except Exception:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500    

    # ------------------------------------API---------------------------------------#
    # Run Processes API Endpoints
    # - /run/sbom_filtering
    # - /run/scoring
    # - /run
    # - /run/progress
    # - /run/generate_csvs
    # - /run/populate_database
    # - /run/refresh_license_info
    # - /run/analysis
    # - /run/redaction
    # - /run/infer_and_update_export


    @app.route("/run/sbom_filtering", methods=["POST"])
    def run_sbom_filtering():
        """Runs sbom filtering process"""
        app.logger.info("Running SBOM Filtering")
        min = request.args.get("min", default=0, type=int)
        if request.is_json:
            sboms_selections = request.get_json()
        else:
            return {"error": f"No sbom filter sent, please send sbom filter json."}, 400
        try:
            filtered_selections = filter_scores_by_min(sboms_selections, min, SCORES)
            save_sbom_filter(filtered_selections, SCORES)
        except:
            error_traceback = traceback.format_exc()
            app.logger.info(f"Error filtering sboms - {error_traceback}")
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... SBOM Filtering Complete")

        return {"message": "SBOM Filtering Successful."}, 200

    @app.route("/run/scoring", methods=["POST"])
    def run_scoring():
        """Runs scoring process"""
        app.logger.info("Running SBOM-QS Scoring")
        try:
            # Delete old scores if they exist
            if os.path.exists(SCORES):
                os.remove(SCORES)
            if os.path.exists(SCORES_XLSX):
                os.remove(SCORES_XLSX)
            # Generate scores
            score_sboms(SBOMS_FOLDER, SCORES, SCORES_XLSX)
        except Exception:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... Scoring Complete")

        return {"message": "Scoring Successful."}, 200

    @app.route("/run", methods=["POST"])
    def run_all():
        """Runs all processes"""
        app.logger.info("Generating CSVs...")
        global PROGRESS
        PROGRESS = (0, "")
        try:
            PROGRESS = (30, "Generating CSVs...")
            generate_csvs()
            PROGRESS = (50, "...CSVs Generated.")
        except Exception as e:
            PROGRESS = (100, "Error Occured.")
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... CSV Generation Complete.")

        app.logger.info("Generating Database...")
        try:
            PROGRESS = (55, "Populating Database...")
            populate_database()
            PROGRESS = (75, "...Database Populated.")
        except Exception as e:
            PROGRESS = (100, "Error Occured.")
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... Database Generation Complete.")

        app.logger.info("Running Analysis ...")
        try:
            PROGRESS = (80, "Running Analysis...")
            analysis()
            PROGRESS = (100, "...Analysis Complete.")
        except Exception as e:
            PROGRESS = (100, "Error Occured.")
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500

        app.logger.info("... Analysis Complete.")
        PROGRESS = (None, "No Process In Progress.")
        return {"message": "Running All Functions Successful."}, 200

    @app.route("/run/progress", methods=["GET"])
    def run_progress():
        """Returns progress on the /run process endpoint"""
        global PROGRESS
        percent, message = PROGRESS
        return {"message": message, "value": percent}, 200

    @app.route("/run/generate_csvs", methods=["POST"])
    def run_generate_csvs():
        """Runs generate csvs process"""
        app.logger.info("Running Generate CSVs...")
        try:
            ## TODO: add the file renaming here until we implement the validation step
            generate_csvs()
        except Exception as e:
            app.logger.error(f"Error occured during CSV generation - {e}")
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... CSV Generation Complete.")

        return {"message": "CSV Generation Successful."}, 200

    @app.route("/run/populate_database", methods=["POST"])
    def run_populate_database():
        """Runs populate database process"""
        app.logger.info("Generating Database")
        try:
            populate_database()
        except Exception as e:
            app.logger.error(f"Error occured during database population - {e}")
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... Database Generation Complete.")

        return {"message": "Database Population Successful."}, 200

    @app.route("/run/refresh_license_info", methods=["POST"])
    def run_license_refresh():
        """Runs license refresh process"""
        app.logger.info("Running Refresh License Information")
        try:
            refresh_license_info(DATABASE)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... License Information Refresh Complete.")

        return {"message": "License Refresh Successful."}, 200

    @app.route("/run/analysis", methods=["POST"])
    def run_analysis():
        """Runs analysis process"""
        app.logger.info("Running Analysis")
        try:
            analysis()
        except Exception:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500
        app.logger.info("... Analysis Complete")

        return {"message": "Analysis Successful."}, 200

    @app.route("/run/redaction", methods=["POST"]) 
    def run_redaction():
        """Runs redaction process"""
        app.logger.info("Running Redaction")

        # Selected preset from hidden field, fallback to default
        policy_name = (request.form.get("preset", "") or "").strip()

        rules_json = request.form.get("rules_json", "[]")
        try:
            rules = json.loads(rules_json)
        except Exception:
            return {"error": "Invalid rules_json"}, 400

        try:
            if not policy_name:
                # No policy selected. Always run with ephemeral policy built from UI rules.
                # If rules == [], build_policy_from_rules returns an empty policy, resulting in no-op redaction.
                policy_dict = build_policy_from_rules(rules)
                policy_tuple = normalize_policy_dict(policy_dict)
                redact(DATABASE, REDACTED_DATA, REDACT_TOML, policy=policy_tuple)
            else:
                if rules:
                    # Named policy selected, but UI rules present, apply ephemeral overrides
                    policy_dict = build_policy_from_rules(rules)
                    policy_tuple = normalize_policy_dict(policy_dict)
                    redact(DATABASE, REDACTED_DATA, REDACT_TOML, policy_name=policy_name, policy=policy_tuple)
                else:
                    # Named policy selected, no UI rules, use the named policy from TOML
                    redact(DATABASE, REDACTED_DATA, REDACT_TOML, policy_name=policy_name)
                
        except Exception:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500

        app.logger.info("... Redaction Complete")
        return {"message": "Redaction Successful."}, 200

    @app.route("/run/infer_and_update_export", methods=["POST"])
    def run_infer_and_update_export():
        """Runs the infer and update export function."""
        try:
            success = infer_and_update_export(DATABASE)  # Only expect one value
            if success:
                # Automatically re-run analysis after successful inference
                analysis()  # Call your analysis function directly
                return {
                    "message": "Export inference completed and analysis re-run"
                }, 200
            else:
                return {"error": "Export inference failed"}, 500
        except Exception as e:
            app.logger.error(f"Error in infer_and_update_export endpoint: {e}")
            return {"error": str(e)}, 500

    # GET API Endpoints
    # - /get/scores
    # - /get/analysis
    # - /download/zip
    # - /download/database
    # - /download/scores
    # - /download/redacted
    # - /submit/org
    # - /submit/assets
    # - /submit/sboms

    @app.route("/get/scores", methods=["GET"])
    def get_scores():
        """Gets sbom scores."""
        app.logger.info("Getting Scores ...")
        if not any(SBOMS_FOLDER.iterdir()):
            return {"error": f"No SBOMS Loaded."}, 400
        try:
            scores = get_scores_from_file(SCORES)
        except Exception as e:
            app.logger.info(f"Error occured getting scores from file - {e} ")
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500

        app.logger.info("... Score Retrival Complete. ")
        return scores, 200

    @app.route("/get/analysis", methods=["GET"])
    def get_analysis():
        """Gets analysis results."""
        number = request.args.get("question", default=0, type=int)
        if number not in [0, 1, 2, 3, 4]:
            return {"error": "Not a valid question."}, 400

        app.logger.info("Getting Analysis Results ...")
        try:
            with open("results.json", "r") as f:
                rq_results = json.load(f)
            if number == 0:
                rq_results = [
                    result for sublist in rq_results.values() for result in sublist
                ]
            else:
                rq_results = rq_results[str(number)]
        except:
            error_traceback = traceback.format_exc()
            return {"error": f"{error_traceback}"}, 500

        app.logger.info("... Analysis Results Retrival Complete. ")
        return rq_results, 200

    @app.route("/download/zip", methods=["GET"])
    def download_zip():
        """Downloads formatted zip file"""
        if ASSET_DATA.exists():
            assets = pd.read_excel(ASSET_DATA, dtype="object")
            structure = create_structure_from_dataframe(assets)
            create_zip_file(BLANK_SBOM_ZIP.with_suffix(""), structure)
            return send_file(BLANK_SBOM_ZIP, as_attachment=True)
        return {"error": f"No asset list uploaded"}, 500

    @app.route("/download/database", methods=["GET"])
    def download_database():
        """Downloads database as a file"""
        return send_file(DATABASE, as_attachment=True)

    @app.route("/download/scores", methods=["GET"])
    def download_scores():
        """Downloads scores as a xlsx"""
        return send_file(SCORES_XLSX, as_attachment=True)

    @app.route("/download/redacted", methods=["GET"])
    def download_redacted():
        """Downloads redacted as a zip"""
        return send_file(REDACTED_DATA.with_suffix(".zip"), as_attachment=True)

    @app.route("/submit/org", methods=["POST"])
    def submit_org_info():
        """Submits organization information and saves to file."""
        name = request.form.get("org_name")
        country = request.form.get("org_country")
        type = request.form.get("org_type")
        # optional arguments
        continent = request.form.get("org_continent") or None
        size = request.form.get("org_size") or None
        ci_sector = request.form.get("ci_sector") or None

        # Validate the data
        if not name or not country or not type:
            return (
                "Name, Country, and Org Type fields are required!",
                400,
            )  # Return an error if validation fails

        # Create a dictionary to store the data
        form_data = {
            "Organization": name,
            "Country": country,
            "Type": type,
            "Continent": continent,
            "Size": size,
            "CISector": ci_sector,
        }
        # Delete previous org info
        if ORG_DATA.exists():
            ORG_DATA.unlink()
        # Save the data to a JSON file
        with open(ORG_DATA, "w") as json_file:
            json.dump(form_data, json_file, indent=4)

        return {"message": "Organization submittion successful"}

    @app.route("/submit/assets", methods=["POST"])
    def upload_asset_list():
        """Uploads Asset Information."""
        if request.method == "POST":
            assets = None
            if "assets" in request.files:
                assets = request.files["assets"]
                assets.save(ASSET_DATA)
                if not validate_asset_file(ASSET_DATA):
                    ASSET_DATA.unlink()
                else:
                    return {"message": "File upload successful"}, 200
        return {"message": "File upload unsuccessful. Try again."}, 400

    @app.route("/submit/sboms", methods=["POST"])
    def upload_sboms():
        """Uploads SBOM Information."""
        if request.method == "POST":
            sboms = None
            if "sboms" in request.files:
                sboms = request.files["sboms"]
                # TODO: Make sure this is a zip file

                # Remove SBOMS if they exist already
                if SBOMS_FOLDER.exists():
                    shutil.rmtree(SBOMS_FOLDER, ignore_errors=True)
                fullpath = Path("/data") / Path("SBOMS.zip")
                # Remove old zip if it exists
                if fullpath.exists():
                    fullpath.unlink()
                sboms.save(fullpath)
                shutil.unpack_archive(fullpath, Path("/data"))
                # TODO: Validate directory structure
                if not validate_directory_structure(SBOMS_FOLDER):
                    fullpath.unlink()  # Remove uploaded archive file
                    shutil.rmtree(SBOMS_FOLDER)  # Remove unpacked archive
                else:
                    rename_all_sboms(SBOMS_FOLDER)  # Rename SBOMS right after upload
                    return {"message": "File upload successful"}, 200
        return {"message": "File upload unsuccessful. Try again."}, 400

    return app


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)
