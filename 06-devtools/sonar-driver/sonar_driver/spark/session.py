# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

import findspark


class SonarSparkSession:

    def __init__(self, spark_home='/usr/global/tools/lcmon/software_stack/spark/current',
                 sonar_cassandra_session=None,
                 spark_opts={'master': 'local[*]'},
                 java_home='/usr/lib/jvm/java-1.8.0/',
                 trust_store='/etc/pki/ca-trust/extracted/java/cacerts'):

        self.sonar_cassandra_session = sonar_cassandra_session

        # Setup Java, Spark env variables
        os.environ['JAVA_HOME'] = java_home
        os.environ['JAVA_OPTS'] = '-Djavax.net.ssl.trustStore=' + trust_store
        os.environ['SPARK_HOME'] = spark_home

        # Add Cassandra connector to packages
        if sonar_cassandra_session is not None:
            existing_packages = spark_opts['packages'] + ',' if 'packages' in spark_opts else ''
            spark_opts['packages'] = existing_packages + 'com.datastax.spark:spark-cassandra-connector_2.11:2.3.0'

        # Setup Spark submit arguments
        pyspark_submit_args = [
            "--" + key + " " + value for (key, value) in spark_opts.items()
        ]

        os.environ['PYSPARK_SUBMIT_ARGS'] = ' '.join(pyspark_submit_args) + " " + "pyspark-shell"

        # Now that they are set, find and initialize Spark
        findspark.init()

        # And finally we can import from pyspark
        from pyspark.sql import SparkSession

        # Build Spark session
        spark_session_builder = SparkSession.builder.appName('SonarSparkSession')

        if sonar_cassandra_session is not None:
            spark_session_builder = (
                spark_session_builder
                    .config('spark.cassandra.connection.host', sonar_cassandra_session.hosts_string)
                    .config('spark.cassandra.auth.username', sonar_cassandra_session.username)
                    .config('spark.cassandra.auth.password', sonar_cassandra_session.token)
            )

        self.spark_session = spark_session_builder.getOrCreate()
