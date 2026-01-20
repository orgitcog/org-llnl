# Copyright 2018 Lawrence Livermore National Security, LLC and other
# sonar-driver Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

import re
import random
import nltk
from nltk.corpus import words


def extract_nodelist(nodelist_str):
    if '[' not in nodelist_str:
        return [nodelist_str]

    r = re.compile('(.*)\[(.*)\]')
    s = r.search(nodelist_str)
    cluster = s.group(1)
    nodeid_ranges = s.group(2)

    def extract_range(range_str):
        rrange = range_str.split('-')
        if len(rrange) > 1:
            return list(map(lambda i: cluster + str(i), range(int(rrange[0]), int(rrange[1]) + 1)))
        else:
            return [cluster + range_str]

    node_ranges = [extract_range(noderange_str) for noderange_str in nodeid_ranges.split(',')]

    return [node for node_range in node_ranges for node in node_range]


extract_nodelist_udf = udf(extract_nodelist, StringType())


def extract_runtime(runtime_str):
    runtime = 0
    elems = runtime_str.split(' ')
    if len(elems) > 1:
        runtime += int(elems[0]) * 24 * 60 * 60
        t = elems[1]
    else:
        t = elems[0]
    telems = t.split(':')
    runtime += int(telems[0]) * 60 * 60 + int(telems[1]) * 60 + int(telems[2])

    return runtime


extract_runtime_udf = udf(extract_runtime, StringType())


nltk.download('words')
english_words = words.words()
salt = str(random.SystemRandom().random())


def anonymize(s):
    random.seed(str(s) + salt)
    return random.choice(english_words)


anonymize_udf = udf(anonymize, StringType())
