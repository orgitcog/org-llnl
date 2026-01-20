![Tests](https://github.com/LLNL/pySigma-pipeline-wintap/actions/workflows/test.yml/badge.svg)
![Status](https://img.shields.io/badge/Status-pre--release-orange)

# pySigma Wintap Pipeline

This is the wintap pipeline for pySigma. It provides the following processing pipelines in `sigma.pipelines.wintap`:

* wintap_pipeline: wintap pipeline to convert windows logs to wintap data model for use with the DuckDB backend

It supports the following output formats:

* default: DuckDB SQL syntax

This pipeline is currently maintained by:

* [Lindsey Whitehurst](https://github.com/LLNL/)

## Usage

See the examples directory for ways to use the pipeline with your wintap datasets.

> [!NOTE]  
> This pipeline will create queries for views not in the default wintap output. These views can be created using the following:

```sql
create or replace table joined_process as
select p.*, pp.args as parent_command_line, pp.process_name as parent_process_name,
[p.file_md5, p.file_sha2] as hashes
from process as p
join process as pp on p.parent_pid_hash=pp.pid_hash and p.daypk=pp.daypk
```

```sql
create or replace table sigma_process_image_load as
select p.process_path, p.args, ri.md5 hashes, i.*
from process_image_load i
join raw_imageload ri on i.pid_hash=ri.pidhash and i.filename=ri.filename and i.daypk=ri.daypk
join process p on i.pid_hash=p.pid_hash and i.daypk=p.daypk
```

```sql
create or replace table sigma_process_registry as
select p.process_path, r.*
from process_registry as r
join process as p on r.pid_hash=p.pid_hash and r.daypk=p.daypk
```

```sql
create or replace table sigma_process_net_conn as
select p.process_path, pnc.*
from process_net_conn as pnc
join process as p on pnc.pid_hash=p.pid_hash and pnc.daypk=p.daypk
```

## Release

LLNL-CODE-837816
