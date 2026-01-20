# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import requests

from sonar_driver.print_utils import pretty_print


class KafkaConnectSession:

    def __init__(self,
                 kafka_rest_url='localhost',
                 kafka_connect_port=8083,
                 kafka_schema_registry_port=8081,
                 debug=False,
                 dry=False):
        self.requests_session = requests.Session()
        self.debug = debug
        self.dry = dry
        self.kafka_rest_url = kafka_rest_url
        self.kafka_connect_port = kafka_connect_port
        self.kafka_schema_registry_port = kafka_schema_registry_port

    def request(self, command, sub_url, port=None, json=None, expected_status_code=201):

        if port is None:
            port = self.kafka_connect_port

        kafka_rest_endpoint = self.kafka_rest_url + ":" + str(port)

        request = requests.Request(
            command,
            kafka_rest_endpoint + sub_url,
            json=json)
        prepared_request = request.prepare()

        if self.dry or self.debug:
            pretty_print(request.__dict__, title="Connector HTTP Request")
        if not self.dry:
            response = self.requests_session.send(prepared_request)
            if self.debug:
                pretty_print(response.json(), title="Connector HTTP Response")
            if response.status_code != expected_status_code:
                raise Exception("Error: status code {} != expected status code {}! Run with -g/--debug to see server response".format(response.status_code, expected_status_code))
            return response

    def install_connector(self, connector_json):
        return self.request('POST', '/connectors/', json=connector_json, expected_status_code=201)

    def uninstall_connector(self, connector_name):
        return self.request('DELETE', '/connectors/' + connector_name, expected_status_code=204)

    def get_connectors(self):
        return self.request('GET', '/connectors', port=self.kafka_connect_port, expected_status_code=200)

    def get_schemas(self):
        return self.request('GET', '/subjects', port=self.kafka_schema_registry_port, expected_status_code=200)

    def get_connector_status(self, connector):
        return self.request('GET', '/connectors/' + connector + '/status', port=self.kafka_connect_port, expected_status_code=200)

    def get_connector_config(self, connector):
        return self.request('GET', '/connectors/' + connector + '/config', port=self.kafka_connect_port, expected_status_code=200)

    def set_connector_config(self, connector, config_json):
        return self.request('PUT', '/connectors/' + connector + '/config', port=self.kafka_connect_port, json=config_json, expected_status_code=200)
