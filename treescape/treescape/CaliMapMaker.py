# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT


class CaliMapMaker:

    def __init__(self):
        self.children_map = {}

    """
      path should look like this: an array of strings.
      ['main', 'lulesh.cycle', 'LagrangeLeapFrog', 'LagrangeElements', 'ApplyMaterialPropertiesForElems']
    """

    def make(self, array):

        # print(array)
        # print('--------------------')
        # print(self.children_map)

        for i in range(len(array) - 1):
            parent = array[i]
            child = array[i + 1]
            if parent in self.children_map:
                if child not in self.children_map[parent]:
                    self.children_map[parent].append(child)
            else:
                self.children_map[parent] = [child]

    def getChildrenMap(self):
        return self.children_map
        # return {'main': ['lulesh.cycle'], 'lulesh.cycle': ['TimeIncrement', 'LagrangeLeapFrog'], 'LagrangeLeapFrog': ['LagrangeNodal', 'LagrangeElements', 'CalcTimeConstraintsForElems'], 'CalcTimeConstraintsForElems': [], 'LagrangeElements': ['CalcLagrangeElements', 'CalcQForElems', 'ApplyMaterialPropertiesForElems'], 'ApplyMaterialPropertiesForElems': ['EvalEOSForElems'], 'EvalEOSForElems': ['CalcEnergyForElems'], 'CalcEnergyForElems': [], 'CalcLagrangeElements': ['CalcKinematicsForElems'], 'CalcKinematicsForElems': [], 'CalcQForElems': ['CalcMonotonicQForElems'], 'CalcMonotonicQForElems': [], 'LagrangeNodal': ['CalcForceForNodes'], 'CalcForceForNodes': ['CalcVolumeForceForElems'], 'CalcVolumeForceForElems': ['IntegrateStressForElems', 'CalcHourglassControlForElems'], 'CalcHourglassControlForElems': ['CalcFBHourglassForceForElems'], 'CalcFBHourglassForceForElems': [], 'IntegrateStressForElems': [], 'TimeIncrement': []}
