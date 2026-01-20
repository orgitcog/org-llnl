import sys
import time

sys.path.append("../python")
sys.path.append("../viz")

from MyTimer import MyTimer
from StackedLine import StackedLine
from CaliReader import CaliReader
from TreeScapeModel import TreeScapeModel

m = MyTimer("start")
# your script here
caliReader = CaliReader("launchdate")

mod = TreeScapeModel(caliReader)
mod.setDrillLevel(["main"])

sl = StackedLine()
sl.render(model=mod, drill_level=["LagrangeLeapFrog"])
m.mark("finish render")
m.print()
