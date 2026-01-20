"""
this is the minimal priithon startup file
it imports all standard Priithon modules into __main__
it imports PriConfig as _priConfig
it fixes the display-hook and execs the Priithon RC file
and - in case of GUI mode (i.e. if sys.app is defined) - it
  sets (default) autosave path 
  fixes GUI exception hook
  and connects a popup menu to the display hook (if PriConfig wants it)
"""
from __future__ import absolute_import
__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

import __main__, sys

exec "from Priithon.all import *" in __main__.__dict__
exec "import Priithon.PriConfig as _priConfig" in __main__.__dict__

if hasattr(sys, 'app') and sys.app: # GUI mode
    __main__.Y._fixGuiDisplayHook()

    __main__.Y._setAutosavePath()
    __main__.Y._fixGuiExceptHook()

    import wx
    if '__WXMAC__' in wx.PlatformInfo:
        # macs tends to have this OpenGL float-texture bug
        # (all images (with pixelvals large compared to 1 appear black)
        # this does not catch the case when Priithon runs
        # remotely on Linux but uses a OSX X-display
        __main__.Y._bugXiGraphics()
    else:  # 20111103 before glutInit was also called on Macs
        __main__.Y._glutInit(argv=sys.argv)
else:
    __main__.U._fixDisplayHook()

#?? del sys
__main__.U._execPriithonRunCommands()

#20051117-TODO: CHECK if good idea  U.naSetArrayPrintMode(precision=4, suppress_small=0)
