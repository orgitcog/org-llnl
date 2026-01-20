# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT


class GraphTraverseModel:
    def __init__(self, th_ens, profiles):
        self.th_ens = th_ens
        self.profiles = profiles
        self.makeMappings()

    def makeMappings(self):

        self.idxByChild = {}
        self.idxByParent = {}

        for i, mynode in enumerate(self.th_ens.graph.traverse()):

            name = mynode.frame["name"]
            childrenNames = []

            for index, childNode in enumerate(mynode.children):
                childName = childNode.frame["name"]
                childrenNames.append(childName)

                self.idxByChild[childName] = name

            self.idxByParent[name] = childrenNames

    #  needed by JS side to traverse tree
    def getParentToChildMapping(self):
        return self.idxByParent

    def getChildToParentMapping(self):
        return self.idxByChild

    #  useful for Python side.
    def getChildrenNamesFor(self, parent):

        childrenNames = []

        for i, mynode in enumerate(self.th_ens.graph.traverse()):

            if mynode.frame["name"] == parent:
                # print("  children=", mynode.children)
                # print("  parents=", mynode.parents)

                for index, childNode in enumerate(mynode.children):

                    childName = childNode.frame["name"]
                    childrenNames.append(childName)

        # print(childrenNames)
        return childrenNames
