# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from sonar_driver.print_utils import pretty_print
from sonar_driver.kafka_connect.session import KafkaConnectSession

def uninstall_connectors(
        connectors,
        kafka_rest_url='localhost',
        kafka_rest_port=8083,
        debug=False,
        dry=False):

    # Initialize Kafka Connect REST session
    kafka_connect_session = KafkaConnectSession(
        kafka_rest_url,
        kafka_rest_port,
        debug, 
        dry
    )

    # Issue uninstall for each
    for connector in connectors:
        kafka_connect_session.uninstall_connector(connector)

    # Get remaining connectors
    connectors_response = kafka_connect_session.get_connectors()
    if not dry:
        connectors = connectors_response.json()
    else:
        connectors = []
    pretty_print(connectors, title="Connectors remaining")


def uninstall_all_connectors(
        kafka_rest_url='localhost',
        kafka_rest_port=8083,
        debug=False,
        dry=False):

    # Initialize Kafka Connect REST session
    kafka_connect_session = KafkaConnectSession(
        kafka_rest_url,
        kafka_rest_port,
        debug, 
        dry
    )

    # Get connectors to uninstall
    connectors_response = kafka_connect_session.get_connectors()
    if not dry:
        connectors = connectors_response.json()
    else:
        connectors = ['all', 'of', 'them']

    uninstall_connectors(connectors, kafka_rest_url, kafka_rest_port, debug, dry)
    

