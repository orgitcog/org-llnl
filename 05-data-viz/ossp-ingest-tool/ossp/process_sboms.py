# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import json, pyjq
from loguru import logger
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, cast
import re


def find_files(directory: Path, exclude_dir=True) -> list:
    """Given a directory, find the files that end with '.json' and return
    a list with those file paths.

    Args:
        directory (Path): path to the files to be listed
        exclude (bool, optional): flag to exclude the 'fw_not_in_use' folder from processing.
            Defaults to True.

    Returns:
        list: list of paths to files.
    """
    exclude = "fw_not_in_use"
    files_found = []

    for file_path in directory.rglob("*.json"):
        # If exclude_dir is True, skip files whose parent directories contain the excluded name.
        if exclude_dir and exclude in file_path.parts:
            continue
        files_found.append(file_path)

    return files_found

def is_empty_json_file(path: Path, check_whitespace: bool = True) -> bool:
    """
    Returns True if the file does not exist, is zero bytes, or only contains whitespace.

    Args: 
        path (Path): path to the file
        check_whitespace (bool): defaults to true 

    Returns:
        bool: True if the file is empty. False if file contains data.
    """
    try:
        if not path.exists():
            return True
        if path.stat().st_size == 0:
            return True
        if check_whitespace:
            # Read a small chunk to avoid loading large files into memory
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                chunk = f.read(4096)
            return chunk.strip() == ""
        return False
    except Exception as e:
        logger.error(f"Error while checking if file is empty {path}: {e}")
        return True


def normalize_provider(raw: Any) -> str:
    """
    Given an input, normalize it for known vendors and return the normalized string.

    Args:
        raw (Any): input that contains the SBOM provider

    Returns:
        str: normalized SBOM vendor string 
    """
    # Accept list or string values
    if isinstance(raw, list):
        raw = " ".join(map(str, raw))
    s = str(raw or "").strip()
    # step below is specific to NetRise SBOMs
    s = s.replace("Tool:", "").replace("tool:", "").strip()
    # Normalize to known keys for get_jq
    sl = s.lower()
    if "netrise" in sl:
        return "NetRise"
    if "finite state" in sl:
        return "Finite State"
    if "docker" in sl or "scout" in sl:
        return "docker-scout"
    return "na"

def get_sbom_metadata(full_path: Path) -> Tuple[str, str, str]:
    """Given an sbom, extract information about the bomFormat, bomProvider, and bomTimestamp.

    Args:
        full_path (Path): path to an sbom.

    Returns:
        tuple(str, str, str): Returns a tuple with the bomFormat, bomProvider, and bomTimestamp.
    """
    if not full_path.exists():
        raise FileNotFoundError(f"File does not exist: {full_path}")
    
    if is_empty_json_file(full_path):
        logger.error(f"SBOM file is empty or whitespace only: {full_path}")
        return ("na", "na", "na")

    # Default values if extraction fails
    bomFormat = "na"
    bomProvider = "na"
    bomTimestamp = "na"

    try:  # open the file
        with full_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # if not throw a json decode error
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {full_path}: {e}")
        return (bomProvider, bomFormat, bomTimestamp)
    except Exception as e:
        logger.error(f"Unexpected error while loading JSON from {full_path}: {e}")
        return (bomProvider, bomFormat, bomTimestamp)

    # Figure out the bomFormat
    if "bomFormat" in data:
        bomFormat = data["bomFormat"]
    elif "spdxVersion" in data:
        bomFormat = data["spdxVersion"]

    if bomFormat == "CycloneDX":
        metadata = data.get("metadata", {})

        # Schema 1: metadata.tools is a dict containing "services"
        if isinstance(metadata.get("tools"), dict) and "services" in metadata["tools"]:
            services = metadata["tools"].get("services", [])
            if services and isinstance(services, list):
                first_service = services[0]
                provider_info = first_service.get("provider", {})
                bomProvider = provider_info.get("name", "na")
            bomTimestamp = metadata.get("timestamp", "na")

        # Schema 2: metadata has a supplier key
        elif "supplier" in metadata and isinstance(metadata["supplier"], dict):
            supplier = metadata["supplier"]
            bomProvider = supplier.get("name", "na")
            # Use metadata timestamp if available; fallback to a top-level timestamp if present
            bomTimestamp = metadata.get("timestamp", data.get("timestamp", "na"))

        # Schema 3: metadata.authors is a list (Finite State)
        elif isinstance(metadata.get("authors"), list):
            supplier = metadata["authors"]
            if supplier:
                first_author = supplier[0]
                bomProvider = first_author.get("name", "na")
            # Timestamp may not exist in this schema; if so, check metadata
            bomTimestamp = metadata.get("timestamp", "na")

        # Schema 4: metadata.tools is a list
        elif isinstance(metadata.get("tools"), list):
            tools = metadata["tools"]
            if tools:
                first_tool = tools[0]
                bomProvider = first_tool.get("name", "na")
            # Timestamp may not exist in this schema; if so, check metadata
            bomTimestamp = metadata.get("timestamp", "na")

    elif "SPDX" in bomFormat:
        ci = data.get("creationInfo") or {}
        bomProvider = ci.get("creators", "na")
        bomTimestamp = ci.get("created", "na")

    else:
        logger.error(f"Unrecognized metadata schema in file: {full_path}")

    return (bomProvider, bomFormat, bomTimestamp)

def sanitize_for_filename(s: str) -> str:
    """
    Given a string, pull out the base file name.

    Args:
        s (str): string that is the name of a file.

    Returns:
        str: string of file name without file extensions.
    """
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(s or ""))

def rename_sbom(bomProvider, bomFormat, bomTimestamp, file_path: Path) -> None:
    """Given an sbom and its bom information, rename the file according to the following
    schema:
            <filename>.<provider full name>.<SBOM format>.<date>.json

    Args:
        bomProvider (str): name of the sbom provider.
        bomFormat (str): sbom format, typically cyclonedx or spdx.
        bomTimestamp (str): sbom creation timestamp.
        file_path (Path): path to sbom.

    Returns:
        Renames the sbom to conform to a standard for easier processing.
    """
    file_path_obj = Path(file_path)
    directory_path = file_path_obj.parent
    original_name = file_path_obj.name

    # Skip renaming empty or missing files
    try:
        if is_empty_json_file(file_path_obj):
            logger.warning(f"Skipping rename of empty SBOM: {file_path_obj}")
            return
    except Exception as e:
        logger.error(f"Could not process file {file_path_obj}: {e}")
        return

    # Clean metadata
    provider_clean = normalize_provider(bomProvider)
    format_clean = str(bomFormat or "unknown").strip()
    timestamp_clean = sanitize_for_filename(bomTimestamp)

    # Base firmware name without extension
    firmware_name = file_path_obj.stem
    new_filename = f"{firmware_name}.{provider_clean}.{format_clean}.{timestamp_clean}.json"
    new_file_path = directory_path / new_filename

    # Avoid unnecessary rename if name is already correct
    if new_file_path.name != original_name:
        try:
            file_path_obj.rename(new_file_path)
            logger.debug(f"\tFile renamed to: {new_file_path}")
        except Exception as e:
            logger.error(f"Failed to rename SBOM file {file_path_obj} - {e}")
    else:
        logger.debug(f"\tFile name not changed - {original_name}")


def process_sbom(full_path: Path, csv_path: Path) -> None:
    """
    Given a path to an SBOM file, extracts the provider information, parses the JSON with jq,
    and writes the pertinent output to CSV files.

    Args:
        full_path (Path): Path to the SBOM file.
        csv_path (Path): Path to store the CSV output.
    """
    # Check that full_path is a file
    if not full_path.is_file():
        logger.error(f"SBOM file does not exist: {full_path}")
        return
    
    if is_empty_json_file(full_path):
        logger.error(f"SBOM file is empty or whitespace only, skipping: {full_path}")
        return

    try:
        with full_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading/parsing JSON from {full_path} - {e}")
        return

    try:
        sbom_provider = extract_sbom_provider_string(full_path)
    except Exception as e:
        logger.error(
            f"Error extracting SBOM provider from filename {full_path.name} - {e}"
        )
        return
    
    # Fallback to 'na' for anything unknown
    if sbom_provider not in ["NetRise", "Netrise", "docker-scout", "Finite State", "na"]:
        logger.debug(f"Unknown provider '{sbom_provider}' for {full_path.name}, using 'na'")
        sbom_provider = "na"

    # Get the proper jq syntax based on SBOM provider
    if sbom_provider in ["NetRise", "Netrise", "docker-scout", "Finite State", "na"]:
        try:
            jq_components, jq_dependencies, jq_sbom_info = get_jq(sbom_provider)
        except Exception as e:
            logger.error(
                f"Error retrieving jq queries for provider {sbom_provider} - {e}"
            )
            return

        try:
            if sbom_provider == "docker-scout":
                # There is no dependency information in the sbom for docker-scout
                processed_components = pyjq.all(jq_components, data)
                processed_sbom_info = pyjq.all(jq_sbom_info, data)

                write_to_output_file(full_path, processed_components, csv_path)
                write_to_output_file(full_path, processed_sbom_info, csv_path, ".ver")
            else:
                processed_components = pyjq.all(jq_components, data)
                processed_dependencies = pyjq.all(jq_dependencies, data)
                processed_sbom_info = pyjq.all(jq_sbom_info, data)

                write_to_output_file(full_path, processed_components, csv_path)
                write_to_output_file(
                    full_path, processed_dependencies, csv_path, ".dep"
                )
                write_to_output_file(full_path, processed_sbom_info, csv_path, ".ver")
        except Exception as e:
            logger.error(
                f"Error processing jq query results for {full_path.name} with provider {sbom_provider} - {e}"
            )
    else:
        logger.error(
            f"SBOM in improper format -- provider not specified in file {full_path.name}"
        )


def extract_sbom_provider_string(full_path: Path) -> str:
    """Given the full path which includes an sbom filename, extract the
    sbom provider information.

    Args:
        full_path (Path): path to sbom

    Returns:
        str: string containing sbom provider initials.
    """
    try:
        bomProvider, _ , _ = get_sbom_metadata(full_path)
        return normalize_provider(bomProvider)
    except Exception as e:
        logger.error(f"Could not extract provider from JSON for {full_path.name} - {e}")
        return "na"


def get_jq(sbom_provider: str) -> Tuple[str, Optional[str], str]:
    """Given an sbom provider string, return the appropriate jq syntax.

    Args:
        sbom_provider (str): string with sbom provider name

    Returns:
        Tuple[str, Optional[str], str]: jq syntax as a string for sbom_info, components, and dependencies
    """
    provider_to_jq = {
        "NetRise": {
            "sbom_info": '[.serialNumber // "", .metadata?.timestamp // "", .version // "", .metadata?.component?."bom-ref" // "", .metadata?.component?.name // "", .metadata?.component?.version // "", .metadata?.component?.type // "", .metadata?.tools?.services[0]?.name // "", .metadata?.tools?.services[0]?.provider?.name // "", .metadata?.tools?.services[0]?.version] | @csv',
            "components": '([(.metadata.component?.name // ""), (.metadata.component?.version // ""), "", "", (.metadata.component?.type // ""), "", "", "", (.metadata.component?."bom-ref" // ""), "", "", "", "" ] | @csv), (.components | .[] | [.name, .version, if .licenses !=null then (.licenses | map(.license.name) | join(",")) else "N/A" end, if .externalReference != null then (( .externalReferences[] | if .url != null then .url else "N/A" end) // "N/A" ) else "N/A" end, .type, .copyright, .cpe, .purl, ."bom-ref", .description, .copyright, .supplier.name, ((.properties // []) | map(select(.name == "netrise:identification:autorun") | .value) | if length > 0 then .[0] else "N/A" end ) ] | @csv)',
            "dependencies": ".dependencies[]? |  [.ref, .dependsOn[]?] | @csv",
        },
        "Netrise": {
            "sbom_info": '[.serialNumber // "", .metadata?.timestamp // "", .version // "", .metadata?.component?."bom-ref" // "", .metadata?.component?.name // "", .metadata?.component?.version // "", .metadata?.component?.type // "", .metadata?.tools[]?.name // "", .metadata?.tools[]?.vendor // "", .metadata?.tools[]?.version] | @csv',
            "components": '([(.metadata.component?.name // ""), (.metadata.component?.version // ""), "", "", (.metadata.component?.type // ""), "", "", "", (.metadata.component?."bom-ref" // ""), "", "", "", "" ] | @csv), (.components | .[] | [.name, .version, if .licenses !=null then (.licenses | map(.license.name) | join(",")) else "N/A" end, if .externalReference != null then (( .externalReferences[] | if .url != null then .url else "N/A" end) // "N/A" ) else "N/A" end, .type, .copyright, .cpe, .purl, ."bom-ref", .description, .copyright, .supplier.name, "N/A" ] | @csv)',
            "dependencies": ".dependencies[]? |  [.ref, .dependsOn[]?] | @csv",
        },
        "docker-scout": {
            "sbom_info": '[.serialNumber // "", .metadata?.timestamp // "", .version // "", .metadata?.component?."bom-ref" // "", .metadata?.component?.name // "", .metadata?.component?.version // "", .metadata?.component?.type // "", .metadata?.tools[]?.name // "", .metadata?.tools[]?.vendor // "", .metadata?.tools[]?.version] | @csv',
            "components": '.components | .[] | [.name, .version, if .licenses !=null then (.licenses | map(.license.id) | join(",")) else "N/A" end, if .externalReference != null then (( .externalReferences[] | if .url != null then .url else "N/A" end) // "N/A" ) else "N/A" end, .type, .copyright, .cpe, .purl, ."bom-ref", .description, .copyright, .supplier.name, "N/A" ] | @csv',
            "dependencies": None,
        },
        "Finite State": {
            "sbom_info": '[.serialNumber // "", .metadata?.timestamp // "", .version // "", .metadata?.component?."bom-ref" // "", .metadata?.component?.name // "", .metadata?.component?.version // "", .metadata?.component?.type // "", .metadata?.supplier?.name // "", .metadata?.tools[1]?.vendor // "", .metadata?.tools[1]?.version] | @csv',
            "components": '.components | .[] | [.name, .version, if .licenses !=null then (.licenses | map(.license.name) | join(",")) else "N/A" end, if .externalReference != null then (( .externalReferences[] | if .url != null then .url else "N/A" end) // "N/A" ) else "N/A" end, .type, .copyright, .cpe, .purl, ."bom-ref", .description, .copyright, .supplier.name, "N/A" ] | @csv',
            "dependencies": '.dependencies[]? |  [.ref, if .dependsOn != null then  (.dependsOn | join(",")) else "N/A" end ] | @csv',
        },
        "na": {
            "sbom_info": '[.serialNumber // "", .metadata?.timestamp // "", .version // "", .metadata?.component?."bom-ref" // "", .metadata?.component?.name // "", .metadata?.component?.version // "", .metadata?.component?.type // "", .metadata?.tools[]?.name // "", .metadata?.tools[]?.vendor // "", .metadata?.tools[]?.version] | @csv',
            "components": '.components | .[] | [.name, .version, if .licenses !=null then .licenses[].license.expression else "N/A" end, if .externalReference != null then (( .externalReferences[] | if .url != null then .url else "N/A" end) // "N/A" ) else "N/A" end, .type, .copyright, .cpe, .purl, ."bom-ref", .description, .copyright, "N/A" ] | @csv',
            "dependencies": '.dependencies[]? |  [.ref, (.dependsOn | join(",")) ] | @csv',
        },
    }

    result = provider_to_jq.get(sbom_provider)
    if not isinstance(result, dict):
        raise ValueError(f"Invalid sbom provider: {sbom_provider}")

    components = cast(str, result.get("components"))
    dependencies = cast(Optional[str], result.get("dependencies"))
    sbom_info = cast(str, result.get("sbom_info"))

    return (components, dependencies, sbom_info)


def write_to_output_file(
    full_path: Path, jq_output, csv_path: Path, file_ending: Optional[str] = None
) -> None:
    """
    Write the jq output to a CSV file. The CSV file will be in a folder/directory that mirrors
    the folder/directory that the processed SBOM file was in.

    Args:
        full_path (Path): Path to the SBOM file that is to be converted to CSV.
        jq_output (iterable): Iterable output from the jq command to be written into the CSV.
        csv_path (Path): Path to the output folder for the CSV file.
        file_ending (str, optional): Suffix to add to the filename before the .csv extension. Defaults to None.
    """
    # Get the directory of the SBOM file and convert to a Path object
    original_directory = full_path.parent

    try:
        # Convert to parts and find the index of "SBOMs"
        parts = original_directory.parts
        sboms_index = parts.index("SBOMs")
    except ValueError:
        logger.error(
            f"'SBOMs' not found in the directory structure of {original_directory}"
        )
        return

    # Create a clean path by taking everything after "SBOMs"
    clean_parts = parts[sboms_index + 1 :]  # ignore "SBOMs" itself
    # Build new directory path under the provided output csv_path
    new_path = csv_path.joinpath(*clean_parts)

    # Create the new file name by removing the .json suffix and appending the provided file ending and .csv
    orig_file_name = full_path.name
    if orig_file_name.endswith(".json"):
        base_name = orig_file_name[:-5]
    else:
        base_name = orig_file_name

    if isinstance(file_ending, str):
        new_file_name = base_name + file_ending + ".csv"
    else:
        new_file_name = base_name + ".csv"

    # Final full output CSV file path
    new_file_path = new_path.joinpath(new_file_name)

    # Create the new directory if it does not exist
    try:
        new_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {new_path}: {e}")
        return

    # Write the jq output to the CSV file
    try:
        with new_file_path.open("w", encoding="utf-8") as output_file:
            for item in jq_output:
                output_file.write(item)
                output_file.write("\n")
    except Exception as e:
        logger.error(f"Error writing CSV to {new_file_path}: {e}")
        return

    logger.debug(f"CSV output written to: {new_file_path}")


def get_excluded_sboms(SCORES: Path):
    """Helper function to grab sboms that have been excluded via scoring

    Args:
        SCORES (Path): path to scores file.
    """
    sboms_to_exclude = []
    if SCORES.exists():
        with open(SCORES, "r") as f:
            scores = json.load(f)
        for file, _ in scores.items():
            if not scores[file]["selected"]:
                sboms_to_exclude.append(file)
    return sboms_to_exclude


def process_all_sboms(directory: Path, csv_path: Path, SCORES: Path) -> None:
    """Function to process all sboms in a directory.

    Args:
        directory (Path): path to directory with sboms.
        csv_path (Path): path to the directory where the csvs will be stored.
    """
    list_of_files = find_files(directory)
    logger.info(f"\n{len(list_of_files)} file(s) found for processing.\n")

    # Incorporate SBOM Filtering
    sboms_to_exclude = get_excluded_sboms(SCORES)

    for item in list_of_files:
        if (
            str(Path(*item.parts[3:])) not in sboms_to_exclude
        ):  # Process SBOM only if it not excluded
            process_sbom(item, csv_path)


def rename_all_sboms(directory: Path) -> None:
    """Function to rename all sboms in a directory.

    Args:
        directory (Path): path to directory with sboms.
    """
    list_of_files = find_files(directory, exclude_dir=False)
    logger.info(f"\n{len(list_of_files)} file(s) found for processing.\n")
    for count, item in enumerate(list_of_files, start=1):
        logger.info(f"Processing file --> {count}. {item}")

        if is_empty_json_file(item):
            logger.warning(f"Skipping empty SBOM file during rename: {item}")
            continue

        try:
            bomProvider, bomFormat, bomTimestamp = get_sbom_metadata(item)
        except Exception as e:
            logger.error(f"Error decoding JSON from {item}: {e}")
            continue  # Skip this file if metadata extraction fails

        # Only assign bomFormat_clean if bomFormat is valid
        if bomFormat == "CycloneDX":
            bomFormat_clean = "cdx"
        elif bomFormat and bomFormat[:4] == "SPDX":
            bomFormat_clean = "spdx"
        else:
            bomFormat_clean = "na"

        try:
            rename_sbom(bomProvider, bomFormat_clean, bomTimestamp, Path(item))
        except Exception as e:
            logger.error(f"Failed to rename SBOM file {item}: {e}")
