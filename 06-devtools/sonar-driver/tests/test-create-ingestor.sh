#!/usr/bin/env bash

# make self-aware
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

${DIR}/../bin/create_directory_source_connector \
    --debug \
    --topic-name "mytopicname" \
    --file-format json \
    --format-options "{\"option\":\"myoption\"}" \
    --tasks-max 1 \
    --batch-size 40000 \
    --backup \
    "my/ingest/dir" "my/completed/dir" ${DIR}/idstr.avsc

${DIR}/../bin/create_cassandra_sink_connector \
    --debug \
    --dry \
    --cassandra-host "mycassandrahost" \
    --cassandra-username "mycassandrauser" \
    --cassandra-password-file "mycassandrapasswordfile" \
    --tasks-max 1 \
    mytopicname mykeyspace mytable

