# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

"""
Kafka Connector Configuration Class
"""

import json


class ConnectorConfig():

    CONNECTOR_CLASS = "UNDEFINED"
    CONNECTOR_CLASS_KEY = "connector.class"
    TASKS_MAX_KEY = "tasks.max"

    def __init__(self, tasks_max=1):
        self.config_dict = {self.CONNECTOR_CLASS_KEY: self.CONNECTOR_CLASS, self.TASKS_MAX_KEY: tasks_max}

    def __str__(self):
        return str(self.json())

    def json(self):
        return {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in self.config_dict.items()}
