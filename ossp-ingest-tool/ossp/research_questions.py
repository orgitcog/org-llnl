# Copyright 2025 Lawrence Livermore National Security, LLC
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List
import pandas as pd
import matplotlib.pyplot as plt
from flask import url_for
from loguru import logger
import networkx as nx
from jinja2 import Environment, FileSystemLoader
import pyvis
from dataclasses import dataclass
import os
from pathlib import Path

from ossp.database import run_sql_script
from ossp.database import (
    rq1_query,
    rq1a_query,
    rq1b_query,
    rq1c_query,
    rq2_query,
    rq3a_query,
    rq3b1_query,
    rq3b2_query,
    rq3b3_query,
    rq3b4_query,
    rq3b5_query,
    rq3c_query,
    rq3d_query,
    rq4_query,
)
from ossp.database import (
    select_latest_sbomid,
    select_firmwareid,
    select_FirmwareName,
    select_ComponentInfo,
    select_filename_from_sbomid,
)


questions = {
    "RQ1": "Ability to identify all OSS services running on, and all OSS components present within, an OT device.",
    "RQ1a": "Ability to differentiate multiple versions of the same OSS component within each OT device.",
    "RQ1b": "Ability to differentiate running from not-running OSS components.",
    "RQ1c": "Ability to differentiate based on the originator of the component, because a supplier may have modified it after retrieval from the upstream software source.",
    "RQ2": "Ability to correlate the identity of a single OSS component across multiple OT devices, mitigating common name variations such as differences in capitalization, '-' vs '_', and so on.",
    "RQ3": "Ability to perform subset analysis of OSS components across multiple OT devices.",
    "RQ3a": "Ability to perform subset analysis across OSS libraries, generating density & distribution graphs to identify commonly-used libraries and outliers.",
    "RQ3b1": "Ability to perform subset analysis of a single OSS library by device make/model.",
    "RQ3b2": "Ability to perform subset analysis of a single OSS library by CI sector.",
    "RQ3b3": "Ability to perform subset analysis of a single OSS library by device type.",
    "RQ3b4": "Ability to perform subset analysis of a single OSS library by firmware version.",
    "RQ3b5": "Ability to perform subset analysis of a single OSS library by component version.",
    "RQ3c": "Ability to perform subset analysis by grouping OSS libraries according to programming language, then overlay with RQ4b.",
    "RQ3d": "Ability to perform subset analysis by OSS upstream source, providing insight into degree of modifications performed by suppliers.",
    "RQ4": "Ability to identify dependencies (transitive and direct) of each differentiated OSS library within each OT device, and enable RQ1,2,3 iteratively for dependencies.",
}


def rq1(dbname: Path) -> List[Dict[str, Any]]:
    """Formats result of research question 1 query."""
    results = []
    df = rq1_query(dbname)

    df_counts = df.drop_duplicates("AssetName")
    df_counts = df_counts.head(10)

    if not df_counts.empty:
        ax = df_counts.plot(
            kind="bar",
            x="AssetName",
            y="UniqueExportedComponents",
            title="OSS Components by Prevalence",
            logy=True,
        )
        ax.set(xlabel="Asset Name", ylabel="Unique OSS Components")
        plt.margins(y=0.5)
        for bar in ax.patches:
            height = bar.get_height()
            width = bar.get_width()
            plt.text(bar.get_x() + width / 2, height, height, ha="center")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout so labels fit
        plt.subplots_adjust(
            bottom=0.5
        )  # Increase bottom margin to prevent label cutoff
        plt.savefig("ossp/static/images/rq1.png")
        results.append(
            {
                "title": f"RQ1: {questions['RQ1']}",
                "type": "image",
                "content": url_for("static", filename="images/rq1.png"),
            }
        )

    return results


def rq1a(dbname: Path) -> dict:
    """Formats result of research question 1a query."""
    df = rq1a_query(dbname)
    data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
    return {"title": f"RQ1a: {questions['RQ1a']}", "type": "table", "content": data}


def rq1b(dbname: Path) -> dict:
    """Formats result of research question 1b query."""
    df = rq1b_query(dbname)
    data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
    return {"title": f"RQ1b: {questions['RQ1b']}", "type": "table", "content": data}


def rq1c(dbname: Path) -> dict:
    """Formats result of research question 1c query."""
    df = rq1c_query(dbname, component="zlib")
    data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
    return {"title": f"RQ1c: {questions['RQ1c']}", "type": "table", "content": data}


def rq2(dbname: Path) -> dict:
    """Formats result of research question 2 query."""
    df = rq2_query(dbname, search="zlib")
    data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
    return {"title": f"RQ2: {questions['RQ2']}", "type": "table", "content": data}


def rq3a(dbname: Path) -> List[Dict[str, Any]]:
    """Formats result of research question 3a query."""
    results = []
    df = rq3a_query(dbname)

    # Convert rows to HTML for the table
    data = df.to_html(classes="styled-table dataframe center", index=False)

    try:
        # Create a bar graph
        # Dynamically set figure height based on number of bars
        n_labels = len(df["Component"])
        plt.figure(figsize=(10, 30))
        ax = df.plot(kind="bar", x="Component", y="ComponentCount", width=1.0)
        plt.title("RQ3a: OSS Libraries by Count")
        plt.xlabel("Component")
        plt.ylabel("Component Count")
        plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
        plt.tight_layout()  # Adjust layout so labels fit
        plt.subplots_adjust(
            bottom=0.5
        )  # Increase bottom margin to prevent label cutoff

        # Dynamically set sparse x-axis labeling
        xticklabels = ax.get_xticklabels()
        n_labels = len(xticklabels)
        n_show = 16  # Number of labels you want to show
        interval = max(1, n_labels // n_show)

        for i, label in enumerate(xticklabels):
            # Only show labels at evenly spaced intervals
            if i % interval != 0 and i != n_labels - 1:
                label.set_visible(False)
        ax.set_xticklabels(xticklabels)

        # Save the plot as an image
        plt.savefig("ossp/static/images/rq3a.png")
        plt.close()
    except Exception as e:
        print(f"Error generating plot: {e}")

    results.append(
        {
            "title": f"RQ3a: {questions['RQ3a']}",
            "type": "image",
            "content": url_for("static", filename="images/rq3a.png"),
        }
    )
    tableData = df.head(50).to_html(
        classes="styled-table dataframe center", index=False
    )
    results.append({"title": f"", "type": "table", "content": tableData})
    return results


def rq3b(dbname: Path) -> list:
    """Formats result of research question 2 query."""

    results = []
    dfs = [
        rq3b1_query(dbname, "zlib"),
        rq3b2_query(dbname, "zlib"),
        rq3b3_query(dbname, "zlib"),
        rq3b4_query(dbname, "zlib"),
        rq3b5_query(dbname, "zlib"),
    ]

    for i, df in enumerate(dfs):
        data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
        results.append(
            {
                "title": f"RQ3b{i+1}: {questions[f'RQ3b{i+1}']}",
                "type": "table",
                "content": data,
            }
        )

    return results


def rq3c(dbname: Path) -> dict:
    """Formats result of research question 2 query."""
    df = rq3c_query(dbname)
    data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
    return {"title": f"RQ3c: {questions['RQ3c']}", "type": "table", "content": data}


def rq3d(dbname: Path) -> dict:
    """Formats result of research question 2 query."""
    df = rq3d_query(dbname)
    data = df.head(10).to_html(classes="styled-table dataframe center", index=False)
    return {"title": f"RQ3d: {questions['RQ3d']}", "type": "table", "content": data}


def rq4(dbname: Path) -> list[dict[str, Any]]:
    """Formats result of research question 4 query."""

    @dataclass
    # Code adapted from Joseph Hand at LLNL
    class DisplaySettings:
        fg_color: str = "white"
        bg_color: str = "#222222"
        node_scale_factor: int = 4

    @dataclass
    # Code adapted from Joseph Hand at LLNL
    class TooltipInfo:
        component_name: str
        bomref: str
        component_type: str
        version: str
        author: str
        language: str

        """
        Formats dict instance into newline separated string
        {'component_name': 'main.exe', 'bomref':'AAA-BBB'} -> 'Component Name:\tmain.exe\nBomRef:\tAAA-BBB\n'
        """

        def toStr(self) -> str:
            return "\n".join(
                k.replace("_", " ").title() + "\t" + str(v)
                for k, v in vars(self).items()
            )

    node_color = {
        "background": "#97C2FC",
        "border": "#2B7CE9",
        "highlight": {
            "background": "#00FF00",  # green when selected
            "border": "#00FF00",
        },
    }

    edge_color = {
        "color": "#848484",  # default edge color
        "highlight": "#FF0000",  # red when highlighted
        "hover": "#FF0000",  # optional: red on hover
    }

    node_color = {
        "background": "#97C2FC",
        "border": "#2B7CE9",
        "highlight": {
            "background": "#00FF00",  # green when selected
            "border": "#00FF00",
        },
    }

    edge_color = {
        "color": "#848484",  # default edge color
        "highlight": "#FF0000",  # red when highlighted
        "hover": "#FF0000",  # optional: red on hover
    }

    results: list[dict[str, Any]] = []

    sbomid = select_latest_sbomid(dbname)
    if sbomid is None:
        logger.error("No latest SBOM ID found. Aborting rq4 query.")
        return results

    for id in range(1, sbomid - 1):
        df = rq4_query(dbname, str(id))

        G = nx.DiGraph()
        df["ParentComponentID"] = df["ParentComponentID"].astype(int)
        df["ChildComponentID"] = df["ChildComponentID"].astype(int)

        for index, row in df.iterrows():
            parent = int(row["ParentComponentID"])
            child = int(row["ChildComponentID"])
            parent_is_root = int(
                row["RootNode"]
            )  # 1 indicates a root component, 0 does not TODO: Remove if we no longer need to track the root component

            try:
                result = select_ComponentInfo(dbname, parent)
                if result:
                    p_name, p_bomref, p_type, p_version, p_author, p_language = result
                    tooltip_parent = TooltipInfo(
                        component_name=p_name,
                        bomref=p_bomref,
                        component_type=p_type,
                        version=p_version,
                        author=p_author,
                        language=p_language,
                    )

                result = select_ComponentInfo(dbname, child)
                if result:
                    c_name, c_bomref, c_type, c_version, c_author, c_language = result
                    tooltip_child = TooltipInfo(
                        component_name=c_name,
                        bomref=c_bomref,
                        component_type=c_type,
                        version=c_version,
                        author=c_author,
                        language=c_language,
                    )
            except:
                logger.debug(f"Error selecting component info for {parent} or {child}")
                continue

            # Add the nodes and their corresponding labels
            G.add_node(
                parent, label=p_name, title=tooltip_parent.toStr(), color=node_color
            )
            G.add_node(
                child, label=c_name, title=tooltip_child.toStr(), color=node_color
            )

            # Add the edge between the child and parent components
            G.add_edge(parent, child, color=edge_color)
            # Only add the dependency edge if the parent is a root component.
            # if parent_is_root == 1:
            #     G.add_edge(parent, child)

        pv = pyvis.network.Network(
            directed=True,
            height="100vh",
            width="100vw",
            bgcolor=DisplaySettings.bg_color,
            font_color=DisplaySettings.fg_color,
        )

        # Get the directory where this Python file is located
        current_dir = Path(__file__).parent
        custom_template_dir = current_dir / "templates"

        # Or if templates is in the project root:
        # custom_template_dir = current_dir.parent / "templates"

        custom_env = Environment(loader=FileSystemLoader(str(custom_template_dir)))

        try:
            custom_template = custom_env.get_template("graph_template.html")
            logger.debug("Custom template loaded successfully.")
        except Exception as e:
            logger.debug("Error loading custom template:", e)

        pv.force_atlas_2based()

        pv.options.interaction.selectConnectedEdges = True

        firmwareid = select_firmwareid(dbname, id)
        firmware_name = select_FirmwareName(dbname, firmwareid)
        sbom_parent_file = select_filename_from_sbomid(dbname, id)

        pv.from_nx(G)
        if len(pv.nodes) > 0:
            pv.show_buttons(filter_=["physics"])
            filename = "dependency_graph_sbom_" + str(id) + ".html"
            pv.write_html("ossp/static/images/" + filename, notebook=False)
            results.append(
                {
                    "title": f"{firmware_name}",
                    "subtitle": f"Dependency Graph for SBOMID {id}: {sbom_parent_file} ",
                    "type": "html",
                    "content": url_for("static", filename="images/" + filename),
                }
            )

    return results
