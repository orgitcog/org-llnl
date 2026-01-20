# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import ipywidgets as widgets
import pandas as pd


class DateRangeSlider:

    def __init__(self, start_datetime, end_datetime):

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        dates = pd.date_range(start_datetime, end_datetime, freq='H')

        options = [(date.strftime('%m/%d %H:00'), date) for date in dates]
        index = (0, len(options) - 1)

        selection_range_slider = widgets.SelectionRangeSlider(
            options=options,
            index=index,
            description='',
            orientation='horizontal',
            layout={'width': '100%'}
        )

        selection_range_slider.observe(self.on_change)

        self.widget = selection_range_slider

    def on_change(self, dummy):
        self.start_datetime = self.widget.get_interact_value()[0]
        self.end_datetime = self.widget.get_interact_value()[0]
