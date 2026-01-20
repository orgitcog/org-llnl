# Copyright 2025 Lawrence Livermore National Security, LLC and other
# Thicket Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import random


class ThicketMultiplierStub:

    def __init__(self, th_ens):
        df_arr = []

        length_of = len(th_ens.dataframe)

        print("length_of=" + str(length_of))

        for j in range(2):
            for i in range(length_of):
                one_item = th_ens.dataframe.iloc[i]
                df_arr.append(one_item)

        th_ens = df_arr

    def old_constructor(self, th_ens):
        # Get the first row
        # first_row = df.iloc[[0]]
        length_of = len(th_ens.dataframe)

        print("length_of=" + str(length_of))

        for j in range(2):
            for i in range(length_of):
                first_row = pd.DataFrame(th_ens.dataframe.iloc[i]).transpose()

                # Concatenate the DataFrame with its first row
                th_ens.dataframe = pd.concat(
                    [th_ens.dataframe, first_row], ignore_index=False
                )

        # th_ens.dataframe.repeat(5)

        # print("len2=" + str(len(th_ens.dataframe)))

    def random_float(self, max):
        return random.uniform(0, max)

    def random_int(self, max):
        return random.randint(0, max)
