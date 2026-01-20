# TreeScape

TreeScape is a Jupyter-based visualization tool for performance data, enabling
users to programmatically render graphs. With TreeScape, you can load an
ensemble of [Caliper](https://github.com/LLNL/caliper) performance files and
visualize the collective performance of an application across many runs. This
could involve tracking performance changes over time, comparing the performance
achieved by different users, running scaling studies across time, etc.
TreeScape is the replacement for SPOT, which was web-based.

# Documentation

The notebook below provides some examples of how to interact with TreeScape:
`regression_notebooks/NightlyTestDemo_local.ipynb`.

# Getting Involved

TreeScape is an open-source project, and we welcome contributions from the
community.

You can also start an issue for a [bug report or feature] request(https://github.com/LLNL/treescape/issues/new).

# Setup Localhost environment, for development
1) Clone the repository:
```
git clone https://github.com/LLNL/treescape.git
```

2) Create a virtual environment:
```
python3 -m venv venv
```

3) Activate the virtual environment:
```
source venv/bin/activate
```

4) Install the dependencies:
```
pip install -r requirements.txt
```

5) Run the notebook:
To make it easier to reach your notebooks, launch jupyter notebook from the directory where your notebooks are.  
```
jupyter notebook
```
6) Open Notebook
After jupyter notebook is running, open the notebook `regression_notebooks/NightlyTestDemo_local.ipynb`.
You will need to modify the paths to the caliper files to match your local environment.
You will need to modify the following paths too:
sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/Users/aschwanden1/github/treescape")

7) Run the cells in the notebook.

# Purpose of each Directory
treescape - This is the main directory.  It contains the python code for the project.
scripts - This directory contains misc test scripts that are used to test exportSVG and other features.
regression_scripts - This directory contains scripts that are used to run regression tests.  You can save a baseline for a set of data, and then compare future data to that baseline.
baseline_results - this directory contains the baseline results for the regression tests.
sandbox - This directory contains notebooks that are used for development and testing.  Contains throwaway code that may be useful for debugging.
regression_notebooks - This directory contains notebooks that are used to run regression tests.
licenses - This directory contains the licenses for the project.
js - This directory contains the javascript code for the project.

# Tree Mapping:
for caliReader, you load cali files like so:
r = cr.CaliperReader("/path/to/cali/files")
r.records then gives you the merged set a of all records from all inputs.
  then in CaliReader the make_child_map function creates the mapping by calling mapMaker.make repeatedly on all the paths

For thicket reader, this is how you get trees:
th_ens = tt.Thicket.from_caliperreader(cali_files)

From that I derive a tree.

Multiple cali_files go in and that function, does not choose a tree.  Instead, it constructs a super tree that is the union of all nodes and call paths found in all those files.

All nodes from all callpaths appear.

The tree grows to include everything encountered.

If a node is missing in a specific file, the metric value for that node is NaN or absent.


# Contributions

We welcome all kinds of contributions: new features, bug fixes, documentation edits; it's all great!

To contribute, make a [pull request](https://github.com/LLNL/treescape/compare),
with `main` as the destination branch.

# Authors

TreeScape was created by Pascal Aschwandan and Matthew LeGendre.

# Release

TreeScape is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE), [COPYRIGHT](./COPYRIGHT) files.

`LLNL-CODE-2010855`
