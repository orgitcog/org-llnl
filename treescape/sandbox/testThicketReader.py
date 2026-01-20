import sys
import time

sys.path.append("/Users/aschwanden1/thicket")
sys.path.append("/Users/aschwanden1/min-venv-local/lib/python3.9/site-packages")
sys.path.append("/Users/aschwanden1/github/treescape")
sys.path.append("/Users/aschwanden1/github/treescape/treescape")

import treescape

from treescape.StackedLine import StackedLine
from treescape.ThicketReader import ThicketReader, TH_ens
from treescape.TreeScapeModel import TreeScapeModel

cali_file_loc = "/Users/aschwanden1/datasets/newdemo/test4"

# your script here
th_obj = TH_ens()
th_ens, profiles = th_obj.get_th_ens( cali_file_loc )
thicketReader = ThicketReader(th_ens, profiles, "launchdate")


mod = TreeScapeModel(thicketReader)

sl = StackedLine()
sl.render( mod, drill_level=['main'] )