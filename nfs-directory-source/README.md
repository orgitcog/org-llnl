# NFS Directory Source

Network FileSystems (NFS) provide the benefit of being distributed and mountable on multiple systems simultaneously, 
and in theory this means we can perform distributed ingestion from data residing in NFS files. 
However, doing so efficiently (i.e. in parallel, without corruption or redundancy) requires advanced inter-process 
synchronization.
This Kafka Connect Source uses Zookeeper to coordinate distributed, scalable ingestion from NFS.

Each NFS directory source specifies (at a minimum):
1. The NFS directory from which to ingest (all files discovered recursively within)
2. The schema of all files inside that directory (one schema per directory source)
3. How to manage ingested files (delete or move to a completed directory)
4. The Kafka topic into which to ingest the data

[Building](doc/building.md)

[Testing](doc/testing.md)

[Deploying](doc/deploying.md)

[Troubleshooting](doc/troubleshooting.md)

See [nfs-directory-source](https://github.com/LLNL/nfs-directory-source) for usage documentation.

License
----------------

nfs-directory-source is distributed under the terms of both the MIT license and the
Apache License (Version 2.0). Users may choose either license, at their
option.

nfs-directory-source also contains the following modified versions of source code licensed by Confluent:
* [AvroConverter.java](src/main/java/io/confluent/connect/avro/AvroConverter.java)
* [AvroConverterConfig.java](src/main/java/io/confluent/connect/avro/AvroConverterConfig.java)
* [AvroData.java](src/main/java/io/confluent/connect/avro/AvroData.java)
* [AvroDataConfig.java](src/main/java/io/confluent/connect/avro/AvroDataConfig.java)

All new contributions must be made under both the MIT and Apache-2.0
licenses.

See [LICENSE-MIT](https://github.com/LLNL/nfs-directory-source/blob/master/LICENSE-MIT),
[LICENSE-APACHE](https://github.com/LLNL/nfs-directory-source/blob/master/LICENSE-APACHE),
[COPYRIGHT](https://github.com/LLNL/nfs-directory-source/blob/master/COPYRIGHT), and
[NOTICE](https://github.com/LLNL/nfs-directory-source/blob/master/NOTICE) for details.

``LLNL-CODE-763876``
