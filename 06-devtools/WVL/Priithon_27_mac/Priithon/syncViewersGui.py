from __future__ import absolute_import
import wx

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

from . import usefulX as Y
from . import PriConfig
import numpy as N

class syncViewersGui(Y.guiParams):
    def __init__(gp):
        super(syncViewersGui, gp).__init__()

        gpL = gp.refreshWinList()
        gp.bb=gp._guiBox(gpL, title="sync viewers gui", ret=True)

        Y.registerEventHandler(gp._doOnPreAnyParamChanged, gp.unFollowVids)
        Y.registerEventHandler(gp._doOnPostAnyParamChanged, gp.followVids)
        Y.registerEventHandler(gp.bb.doOnClose, gp.unFollowVids)


    def doRefresh(self, *a):
        # remove all elements
        self.bb.sizer0Vert.DeleteWindows()
        while self.bb.sizer0Vert.Remove(0):
            pass
        self.bb.startNewRow()
        gpL = self.refreshWinList()
        Y.buttonBoxAdd(gpL, self.bb.i)
        self.bb.frame.Layout()
        self.bb.frame.Fit()

    def refreshWinList(gp):
        gpL= []
        for vid,v in enumerate(Y.viewers):
            if v:
                gpL.append( ("l\t%d) "%(vid), "", 0, False, "viewer #%d"%(vid)) )
                varN='txtScale_%d'%vid
                ii = gp._bboxText("", varN, v=gp._paramsVals.get(varN, '1'), newLine=False, 
                                  labelWeight=0, labelExpand=False, textWeight=0, textExpand=False, textWidth=-1, 
                                  tooltip="use this if viewer shows a magnified version of the image\n"+
                                  "format: '<sy> <sx>' or '<scale>'" , regFcn=None, regFcnName=None)
                gpL.extend( ii )
                varN='boolSync_%d'%vid
                ii = gp._bboxBool(v.title, n=varN, v=gp._paramsVals.get(varN, 0),controls="cb", newLine=False, regFcn=None, regFcnName=None, tooltip="sync vid #%d"%vid)
                gpL.extend( ii )
                #gpL.append( ("l\t%s"%(v.title), "") )
                gpL.append( "\n" )

        gpL.append( ("update viewers list", gp.doRefresh, 1, True, "") )
        return gpL

    def unFollowVids(gp, *a):
        vids = [int(k.split("_")[1]) for (k,v) in gp._paramsVals.items() if k.startswith("boolSync") and v]
        #Y.vSyncViewersReset(vids=vids)
        for vid in vids:
            if Y.viewers[vid]:
                Y.vFollowMouse(vid) 

    def followVids(gp, *a):
        vids = [int(k.split("_")[1]) for (k,v) in gp._paramsVals.items() if k.startswith("boolSync") and v]
        global scalesTxts
        crossColor=PriConfig.defaultGfxColor
        scalesTxts = dict([ (vid,gp._paramsVals["txtScale_%d"%vid]) for vid in vids ])
        scales={}
        for vid, scaleTxt in scalesTxts.iteritems():
            scaleTxt=scaleTxt.strip()
            if scaleTxt == '':
                scale = N.array((1,1))
            else:
                sc = scaleTxt.split()
                if len(sc)==1:
                    sc *= 2
                elif len(sc)!=2:
                    wx.Bell()
                    sc = sc[:2]
                try:
                    scale = N.array(map(float,sc))
                except ValueError:
                    scale = N.array((1., 1.))
            scales[vid] = scale

        for vid in vids:
            scaleVid = scales[vid]
            xyVids= [(vi,tuple(scales[vi]/scaleVid)) for vi in vids if vi != vid]
            Y.vFollowMouse(vid, 
                           xyVids,
                           crossColor=crossColor)
        #Y.vSyncViewers(vids=vids, color=(0, 1, 0), syncHist=True, registerDClick=True)
