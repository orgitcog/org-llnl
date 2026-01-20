# Open Source Software Prevalence Ingest Tool (OSSP Ingest Tool)

This project provides a web interface for ingesting, analyzing, and visualizing Software Bill of Materials (SBOM) data with asset data. The data can then be consolodated, redacted, and exported for downstream data aggregation or tool use.

## Getting Started

### Prerequisites

- Docker
- Web Browser

### Running OSS Prevalence Ingest Tool
Build the Docker container:

```bash
docker build -f Dockerfile . -t ossp-server
```

Run the container:

```bash
docker run --rm -p 5000:5000 --name ossp-server ossp-server
```
Now, use a web browser to navigate to [localhost:5000](http://localhost:5000). The OSSP web page should display.

## Typical Workflow

### 1. Ingest SBOM and Asset data
- Upload asset list using provided Excel sheet layout
- Download pre-formatted .zip file containing the needed directory structure
- Populate .zip folder structure with SBOM's provided by external tools
- Upload populated .zip file

### 2. Score SBOM's
- Score SBOMs based on quality and completeness metrics derived from Interlynk's sbomqs
- View detailed subscores for each category
- Deselect SBOMs to ingest based on score thresholds or manually

### 3. Perform data ingestion and analysis
  - Fill in organization information and click Submit
  - Click "Run All Processes" to ingest and analyze the input data. This may take several minutes
  - Optionally, click "Infer Licenses" to infer licenses based on simple component name matching

### 3. Demonstrate Analytics
  - Click Results to view analytical results

  - The tables and graphs display demonstrate that this system can answer the following research questions:
    - RQ1. Ability to identify all OSS services running on, and all OSS components present within, an OT device 
    - RQ1a: Ability to differentiate multiple versions of the same OSS component within each OT device. 
    - RQ1b: Ability to differentiate running from not-running OSS components. 
    - RQ1c: Ability to differentiate based on the originator of the component, because a supplier may have modified it after retrieval from the upstream software source. 
    - RQ2. Ability to correlate the identity of a single OSS component across multiple OT devices, mitigating common name variations such as differences in capitalization, ‘-’ vs ‘_’, and so on. 
    - RQ3. Ability to perform subset analysis of OSS components across multiple OT devices      
    - RQ3a: Ability to perform subset analysis across OSS libraries, generating density & distribution graphs to identify commonly-used libraries and outliers.     
    - RQ3b: Ability to perform subset analysis of a single OSS library, generating density & distribution by CI sector, by device type, by device make/model, and/or by firmware version. 
    - RQ3c: Ability to perform subset analysis by grouping OSS libraries according to programming language, then overlay with RQ4b. 
    - RQ3d: Ability to perform subset analysis by OSS upstream source, providing insight into degree of modifications performed by suppliers. 
    - RQ4. Ability to identify dependencies (transitive and direct) of each differentiated OSS library within each OT device, and enable RQ1,2,3 iteratively for dependencies.  
 
### 4. Redact
- Click Redact to define data redaction policy

### 5. Export
 - Export redacted data as csv files, which is easily ingestible by a downstream aggregator or external tool


## Technologies in use

- Python Flask for the web server
- SQLite for the database
- Bootstrap for the UI
- Various Python libraries: pandas, networkx, pyvis, matplotlib
- SBOMQS fork for sbom scoring

## License

Copyright 2025 Lawrence Livermore National Security, LLC. See the top-level COPYRIGHT file for details.

This software was developed with funding from the Cybersecurity and Infrastructure Security Agency of the U.S. Department of Homeland Security.

OSSP Ingest Tool is distributed under the terms of the MIT license. All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-2011028