# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import json
import subprocess
from pathlib import Path
from loguru import logger
import tempfile
import shutil
import traceback
from uuid import uuid4
from ossp.process_sboms import find_files


def run_sbomqs_on_directory(directory: Path):
    process = subprocess.run(
        [
            "/bin/sbomqs-ossp",
            "score",
            directory,
            "--configpath",
            "/features.yaml",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode == 0:
        logger.debug(f"Successfully ran sbomqs-ossp on directory {directory}")
        return json.loads(process.stdout)
    else:
        logger.error(f"Error running sbomqs-ossp on {directory} - {process.stderr}")
        return None


def get_scores_from_file(score_path: Path) -> dict:
    with open(score_path, "r") as f:
        scores = json.load(f)

    return scores


def process_scores(scores: list, file_mapping: dict) -> dict:
    processed_scores = {}
    for score in scores:
        filename = Path(score["file_name"]).stem
        processed_scores[file_mapping[filename]] = {
            "avg_score": round(score["avg_score"], 2),
            "subscores": [],
            "score_id": filename,
            "selected": True,
        }
        for subscore in score["scores"]:
            del subscore["ignored"]
            del subscore["max_score"]
            subscore["score"] = round(subscore["score"], 2)
            processed_scores[file_mapping[filename]]["subscores"].append(subscore)
    return processed_scores


def format_scores_to_xlsx(scores: dict, xlsx_path: Path):
    subscores = []
    avg_scores = []
    for filename, scores in scores.items():
        avg_scores.append({"filename": filename, "avg_score": scores["avg_score"]})
        for score in scores["subscores"]:
            score["filename"] = filename
            subscores.append(score)

    subscores_df = pd.DataFrame(subscores).loc[
        :, ["filename", "category", "feature", "description", "score"]
    ]
    avg_scores_df = pd.DataFrame(avg_scores)
    with pd.ExcelWriter(xlsx_path) as writer:
        subscores_df.to_excel(
            writer, sheet_name="Subscores", index=False, float_format="%.2f"
        )
        avg_scores_df.to_excel(
            writer, sheet_name="Average Scores", index=False, float_format="%.2f"
        )


def score_sboms(sboms: Path, score_path: Path, score_xlsx: Path):
    files = find_files(directory=sboms)
    file_mapping = {}
    processed_scores = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        for file in files:
            file = Path(file)
            if file.stat().st_size == 0: #skip empty files
                logger.debug(f"{file} is empty, skipping")
                continue
            id = str(uuid4())
            file_mapping[id] = str(Path(*file.parts[3:]))
            shutil.copy2(file, Path(tmp_dir) / id)
        try:
            scores = run_sbomqs_on_directory(Path(tmp_dir))
            if scores is None:
                return
            processed_scores = process_scores(scores["files"], file_mapping)
        except Exception as e:
            logger.error(
                f"SBOMQS Error: failed to run and process sbomqs scoring - {e} - {traceback.format_exc()}"
            )
            raise e
        try:
            if processed_scores:
                with open(score_path, "w") as f:
                    json.dump(processed_scores, f)
                format_scores_to_xlsx(processed_scores, score_xlsx)
        except Exception as e:
            logger.error(f"Pandas Error: failed to format scores into xlsx file - {e}")
            raise e


def filter_scores_by_min(sboms_selections: dict, min: int, scores_path: Path):
    logger.debug(f"Filtering sboms with min={min}.")

    if scores_path.exists():
        with open(scores_path, "r") as f:
            scores = json.load(f)

        filtered_selections = {}
        for name, selected in sboms_selections.items():
            if selected:
                if scores[name]["avg_score"] >= min:
                    filtered_selections[name] = True
                else:
                    filtered_selections[name] = False
            else:
                filtered_selections[name] = False

        return filtered_selections
    else:
        raise FileNotFoundError(
            "Scores path does not exist, run scoring before filtering selections."
        )


def save_sbom_filter(sbom_selections, scores_path: Path):
    logger.debug(f"Saving sbom selections: {sbom_selections}")
    if scores_path.exists():
        with open(scores_path, "r") as f:
            updated_scores = json.load(f)
        for name, selected in sbom_selections.items():
            updated_scores[name]["selected"] = selected
        with open(scores_path, "w") as f:
            json.dump(updated_scores, f)
    else:
        raise FileNotFoundError(
            "Scores have not be calculated, cannot process sbom filter."
        )
