# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

class Connector():
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def json(self):
        return { 
            "name" : self.name, 
            "config" : self.config.json() 
        }
