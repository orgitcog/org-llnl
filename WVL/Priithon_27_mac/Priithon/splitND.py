"""provides the parent frame of Priithon's ND 2d-section-viewer"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

#from .splitNDcommon import *
#SyntaxError: 'import *' not allowed with 'from .'
#<hack>
from . import splitNDcommon
for n in splitNDcommon.__dict__:
    if not n.startswith('_'):
        exec "%s = splitNDcommon.%s" % (n,n)
del n, splitNDcommon
#</hack>


def viewImgFiles(filenames):
    for fn in filenames:
        #see email
        #Re: [wxPython-users] wxFileDropTarget get filename %-encoded (on gtk not on msw)
        #From: Robin Dunn <robin@alldunn.com>
        #To: wxPython-users@lists.wxwidgets.org
        #Date: Dec 1 2004  Wednesday 12:33:49 pm

        if wx.Platform == "__WXGTK__" and wx.VERSION[:2] == (2,4):
            import urllib
            fn = urllib.unquote(fn)

        run(fn, _scoopLevel=2)


def run(img, title=None, size=None, originLeftBottom=None, 
        panel=None, _scoopLevel=1): #20110901 , _filename=None): 
    # originally this function was added just to not get a return value
    """img can be either an n-D image array (n >= 2)
          or a filename for a Fits or Mrc or jpg/gif/... file
          or a tuple of the above to open as mockNDarray
          or a list  of the above to open multiple viewers
       """
    import os
    from .all import Y

    if type(img) is list:
        for i in img:
            run(i, title, size, originLeftBottom, panel, _scoopLevel=2)
        return
    if type(img) is tuple:
        from . import fftfuncs as F
        imgs = tuple(( Y.load(i) if isinstance(i, basestring) else i for i in img ))
        moa = F.mockNDarray(*imgs)
        run(moa, title, size, originLeftBottom, panel, _scoopLevel=2)
        return

    #"filename"
    if isinstance(img, basestring) and os.path.isfile(img):
        fn=img
        p,f = os.path.split(os.path.abspath(fn))
        #print fn, (fn[:6] == "_thmb_"), (fn[-4:] == ".jpg")
        if f[:6] == "_thmb_" and f[-4:] == ".jpg":
            f = os.path.join(p, f[6:-4])
            if os.path.isfile( f ):
                fn = f

        elif f[-4:] == ".txt":
            from .mmviewer import mview
            if size is None:
                return mview(fn)
            else:
                return mview(fn, size=size)
                
        a = Y.load(fn) #20051213
        if a is None:
            return
        #20060824 CHECK  if originLeftBottom is None and \
        #20060824 CHECK     hasattr(a, '_originLeftBottom'):
        #20060824 CHECK      originLeftBottom = a._originLeftBottom
        # if title is None:
        #     import os.path
        #     title = "<%s>" % os.path.basename(fn)

        #self.filename = fn # 20100811
        return run(a, title, size, originLeftBottom=originLeftBottom, 
                   panel=panel, _scoopLevel=2) #20110901 , _filename=fn)
    if title is None:
        # python expression: evaluate this string and use it it as title !
        if isinstance(img, basestring): # title
            try:
                import sys
                fr = sys._getframe(_scoopLevel)
                locs = fr.f_locals
                globs = fr.f_globals
                a = eval(img, globs, locs)
                img,title = a, img
            except ValueError: # HACK: stack not deep enough
                pass
        #eval("Y.view(%s, '%s', %s)" % (img, img, size), locs) # HACK

        else:     # see if img has a name in the parent dictionary - use that as title
            try:
                import sys
                fr = sys._getframe(_scoopLevel)
                vars = fr.f_globals.copy()
                vars.update( fr.f_locals )
                for v in vars.keys():
                    if vars[v] is img:
                        title = v
                        break
            except ValueError: # stack not deep enough
                pass
    if title is None:
        title=''
            
    ss=spv(img, title, size, originLeftBottom, panel)
    #20110901 if _filename is not None:
    #20110901     ss.filename = _filename # 20100811
    
class spv(spvCommon):
    """ 
    "split panel viewer"
    self.hist_arr != None ONLY IF NOT self.img.type() in (na.UInt8, na.Int16, na.UInt16)
    then also   self.hist_max   and   self.hist_min  is set to min,max of number type !!
    and:  self.hist_range = self.hist_max - self.hist_min
    then  call:
    S.histogram(self.img, self.hist_min, self.hist_max, self.hist_arr)
    self.hist.setHist(self.hist_arr, self.hist_min, self.hist_max)
    
    
    otherwise call   self.recalcHist()
    this _should_ be done from worker thread !?
    
    """



##thrd       class ResultEvent(wx.PyEvent):
##thrd           """Simple event to carry arbitrary result data"""
##thrd       
##thrd           def __init__(self, data):
##thrd               wx.PyEvent.__init__(self)
##thrd               self.SetEventType(EVT_RESULT_ID)
##thrd               self.data = data

    def __init__(self, data, title='', size=None, 
                 originLeftBottom=None, panel=None):
        """
        splitter window for single-color viewerer
        combines a "topBox" - zslider, OnMouse info,
        a viewer window
        and a set histogram windows (one for each color)

        if `panel` is None: makes a new frame with "smart" title and given size
        """
        spvCommon.__init__(self)

        # 20070715: what can we do with zeros in zshape - skip slider ?!
        if not isinstance(data, F.mockNDarray):
            data = N.asanyarray(data) # 20060720 - numpy arrays don't have ndim attribute
        if min(data.shape) < 1:
            raise ValueError, "data shape contains zeros (%s)"% (data.shape,)
        

        if not 1 < data.ndim:
            raise ValueError("cannot display %dD data"% data.ndim)


        try:
            _1checkIt = repr(data)   # protect against crash from ""error: copy2bytes: access beyond buffer""
            del _1checkIt
        except:
            raise

        ####self.copyDataIfUnsupportedType(data)
        self.data = data

        self.zshape= self.data.shape[:-2]
        self.zndim = len(self.zshape)
        self.zsec  = [0] * self.zndim
        self.zlast = [0]*self.zndim # remember - for checking if update needed
        #FIMXE: next line should be done by calling helpNewData() instead - see below
        self.img  = self.data[ tuple(self.zsec) ]
        if self.img.dtype.type  in (N.complex64, N.complex128):
            if True: #self.m_viewComplexAsAbsNotPhase: (memo20051128-> viewComplexAsAbsNotPhase in viewer-class
                self.img = N.asarray(abs(self.img), N.float32) # check if this does temp copy
            else:
                #from .all import U
                #data = U.phase(self.m_imgArr.astype(na.float32)
                #not temp copy for type conversion:
                self.img =  N.arctan2(N.asarray(self.img.imag, N.float32),
                                      N.asarray(self.img.real, N.float32))

        self.recalcHist_todo_Set = set()
        from .usefulX import viewers
        n = len( viewers )
        #self.__class__.viewers[ title ] = self
        viewers.append( self )
        self.id = n

        if panel is None:
            panel=self.makeFrame(size) #20110902, title)
            needShow=True
        else:
            needShow=False
            
        #20100806 splitter = wx.SplitterWindow(panel, -1, style=wx.SP_LIVE_UPDATE|wx.SP_3DSASH)
        self.splitter = wx.SplitterWindow(panel, -1, style=wx.SP_LIVE_UPDATE|wx.SP_PERMIT_UNSPLIT)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.upperPanel = wx.Panel(self.splitter, -1)
        self.upperPanel.SetSizer(sizer)
        self.upperPanel.SetAutoLayout(True)

        self.boxAtTop = wx.BoxSizer(wx.HORIZONTAL)

        self.putZSlidersIntoTopBox(self.upperPanel, self.boxAtTop)
        sizer.AddSizer(self.boxAtTop, 0, wx.GROW|wx.ALL, 2)
        
        from . import viewer
        v = viewer.GLViewer(self.upperPanel, self.img, originLeftBottom=originLeftBottom)
        self.viewer = v
        
        self.viewer.Bind(wx.EVT_IDLE, self.OnIdle)

        tlParent = wx.GetTopLevelParent(panel)
        v.m_menu.Insert(v.m_menu.GetMenuItemCount()-1, Menu_autoSetSize, "auto size frame\ta")
        wx.EVT_MENU(tlParent, Menu_autoSetSize, self.OnAutoSizeFrame)
        
        if self.zndim > 0:
            v.m_menu.Append(Menu_ProjMax, "max-project along first axis\tV")
            v.m_menu.Append(Menu_ProjMean, "mean-project along first axis\tShift-V")
            wx.EVT_MENU(tlParent, Menu_ProjMax,  self.OnViewMaxProj)
            wx.EVT_MENU(tlParent, Menu_ProjMean, self.OnViewMeanProj)
            self.addZsubMenu(tlParent)


        menuSub2 = wx.Menu()
        
        from .all import Y
        self.plot_avgBandSize=1
        self.plot_s='-+'
        def OnChProWi(ev):
            i= wx.GetNumberFromUser("each line profile gets averaged over a band of given width",
                                    'width:', "profile averaging width:",
                                    self.plot_avgBandSize, 1, 1000)
            self.plot_avgBandSize=i
            #Y.vLeftClickNone(self.id) # fixme: would be nice if
            #done!?
            Y._plotprofile_avgSize = self.plot_avgBandSize
        def OnSelectSubRegion(ev):
            Y.vLeftViewSubRegion(self.id)
                
        left_list = [('horizontal profile',
                      lambda ev: Y.vLeftHorizProfile(self.id, bandSize=self.plot_avgBandSize, c=self.plot_s)),
                     ('vertical profile',
                      lambda ev: Y.vLeftVertProfile(self.id, self.plot_avgBandSize, c=self.plot_s)),
                     ('any-line-profile',
                      lambda ev: Y.vLeftClickLineProfile(self.id, abscissa='line', s=self.plot_s)),
                     ('any-line-profile over x',
                      lambda ev: Y.vLeftClickLineProfile(self.id, abscissa='x', s=self.plot_s)),
                     ('any-line-profile over y',
                      lambda ev: Y.vLeftClickLineProfile(self.id, abscissa='y', s=self.plot_s)),
                     ('Z-profile',
                      lambda ev: Y.vLeftClickZProfile(self.id, self.plot_avgBandSize, self.plot_s)),
                     ('line measure',
                      lambda ev: Y.vLeftClickLineMeasure(self.id)),
                     ('triangle measure',
                      lambda ev: Y.vLeftClickTriangleMeasure(self.id)),
                     ('mark-cross',
                      lambda ev: Y.vLeftClickMarks(self.id, callFn=None)),
                     ('<clear click-function>',
                      lambda ev: Y.vLeftClickNone(self.id)),
                     ('<clear graphics>',
                      lambda ev: Y.vClearGraphics(self.id)),
                     ('<change profile "width"',
                      lambda ev: OnChProWi(ev)),
                     ('select-view xy-sub-region',
                      lambda ev: OnSelectSubRegion(ev)),
                     ]
        for i in range(len(left_list)):
            itemId = Menu_LeftClickMenu+1+i
            menuSub2.Append(itemId, "%s" % left_list[i][0])
            wx.EVT_MENU(panel, itemId, left_list[i][1])
        v.m_menu.AppendMenu(Menu_LeftClickMenu, "on left click ...", menuSub2)


        v.m_menu_save.Append(Menu_SaveND,    "save nd data stack")
        v.m_menu_save.Append(Menu_AssignND,  "assign nd data stack to var name")
            
        wx.EVT_MENU(panel, Menu_SaveND,      self.OnMenuSaveND)
        wx.EVT_MENU(panel, Menu_AssignND,      self.OnMenuAssignND)
        
        #dt = MyFileDropTarget(self)
        #v.SetDropTarget(dt)
        from . import fileDropPopup
        v.SetDropTarget( fileDropPopup.FileDropTarget(v) )

        
        def splitND_onMouse(x=-1,y=-1, _ev=None): #20080707 ,xyEffVal): #20101112:defaults for doOnPanZoom
            yy,xx = int(round(y)), int(round(x)) # NEW 20080701:  in new coord system, integer pixel coord go through the center of pixel
            try:
                pic_ny, pic_nx = self.img.shape
            except AttributeError:
                return             # can happend when window just got closed (windows)
            if (0<=yy<pic_ny and
                0<=xx<pic_nx):
                xyEffVal = self.img[yy,xx] #20080707
                if N.issubdtype(self.data.dtype, N.integer): #(N.uint8, N.int16, N.uint16, N.int32):
                    vs = "%6d"  %(xyEffVal,)
                elif self.data.dtype == bool:
                    vs = "%4d"  %(xyEffVal,)
                elif PriConfig.viewerShowComplexVals and N.iscomplexobj(self.data):
                    c = self.data[tuple(self.zsec)][yy,xx]
                    vs = "%7.2f+%.2fj" % (c.real, c.imag)
                else:
                    if N.abs(xyEffVal) > .02:
                        vs = "%7.2f"  %(xyEffVal,)
                    else:
                        vs = "%7.2e"  %(xyEffVal,)
            else:
                vs = "---"
                #self.label.SetLabel("xy: %3d %3d  val: %7.2f"%(x,y, xyEffVal))#self.img[y,x]))
            if v.m_scale > 1 and self.showFloatCoordsWhenZoomingIn:
                self.label.SetLabel("%.1fx yx: %5.1f %5.1f  val: %s"%(v.m_scale, y,x, vs))
            elif v.m_scale !=1:
                self.label.SetLabel("%.1fx yx: %3d %3d  val: %s"%(v.m_scale, yy,xx, vs))
            else:
                self.label.SetLabel("yx: %3d %3d  val: %s"%(yy,xx, vs))
        v.doOnMouse.append(splitND_onMouse)
        v.doOnPanZoom.append(splitND_onMouse)

        from . import histogram

        h = histogram.HistogramCanvas(self.splitter, size=(400,0))
        self.hist   = h
        #20070525-black_on_black h.SetCursor(wx.CROSS_CURSOR)
        import weakref  # 20060823
        # 20060823 v.hist4colmap = weakref.proxy( h ) # HACK
        # see viewer.py::updateHistColMap
        v.my_hist   = weakref.proxy( h ) # CHECK 20060823
        h.my_viewer = weakref.proxy( v ) # CHECK 20060823
        v.my_spv    = weakref.proxy( self ) # CHECK 20070823
        h.my_spv    = weakref.proxy( self ) # CHECK 20070823

        def splitND_onBrace(s, gamma=None):
            l,r = s.leftBrace, s.rightBrace
            try:
                if gamma is not None:
                    v.cmgray(gamma)
                v.changeHistogramScaling(l,r)
            except:
                pass
        h.doOnBrace.append(splitND_onBrace)
        #20080707 del splitND_onBrace
        def splitND_onMouseHist(xEff, _ev): #20080707 , bin):
            l,r = h.leftBrace, h.rightBrace
            if self.data.dtype.type in (N.uint8, N.int16, N.uint16, N.int32):
                self.label.SetLabel("I: %6.0f  l/r: %6.0f %6.0f"  %(xEff,l,r))
            else:
                self.label.SetLabel("I: %7.2g  l/r: %7.2g %7.2g"%(xEff,l,r))
        h.doOnMouse.append(splitND_onMouseHist)
        #20080707del splitND_onMouseHist
    
        #from . import seb as S
    
        def splitND_onReload(event=None):
            self.helpNewData()

        v.OnReload = splitND_onReload
        wx.EVT_MENU(v, viewer.Menu_Reload,      splitND_onReload)
        self.OnReload = splitND_onReload
        del splitND_onReload

    
        #self.hist_min, self.hist_min, self.hist_avg, self.hist_dev

        sizer.Add(v, 1,  wx.GROW|wx.ALL, 2)

        if hasattr(self, 'downSizeToFitWindow') and self.downSizeToFitWindow:
            fac = 1./1.189207115002721 # >>> 2 ** (1./4)
            #v.m_scale *= .05 # fac
            s=max(self.img.shape)
            while v.m_scale * s > 600:
                v.m_scale *= fac
        

        #20070809 wx.Yield()
        if needShow:
            panel.Show()
            self.installKeyCommands(panel)
            # ACCEL_CMD:"Cmd" is a pseudo key which is the same as Control for PC and Unix platforms but the special "Apple" (a.k.a as "Command") key on Macs.
            self.keyShortcutTable[ wx.MOD_CMD, ord('W') ] = panel.Close

        #20040317  splitter.SetMinimumPaneSize(20)
        self.splitter.SetMinimumPaneSize(5)
        self.splitter.SetSashGravity(1.0)
        self.splitter.SplitHorizontally(self.upperPanel, h, -50)
        #77 splitter.SplitHorizontally(v, h, -50)

        #import pdb
        #pdb.set_trace()
        self.setupHistArr()
        self.recalcHist(triggeredFromIdle=True)
        #print "debug:", self.mmms
        self.hist.autoFit(amin=self.mmms[0], amax=self.mmms[1])
        #20051128 wx.Yield()
        #v.changeHistogramScaling(self.mmms[0],self.mmms[1])

        #20100325 wx.Yield()
        #20100325 v.center() # center called anyway in OnPain if `m_x0 is None`
        wx.EVT_CLOSE(wx.GetTopLevelParent(panel), self.onClose)
        self.keyShortcutTable[ wx.MOD_SHIFT, 'c' ] = self.onGammaShortcut  # needs normalizeKeyShortcutTable() called afterwards
        self.setDefaultKeyShortcuts()                   # calls normalizeKeyShortcutTable()
        self.setFrameTitle(title) #20110902
        
##thrd           # Set up event handler for any worker thread results
##thrd           def EVT_RESULT(win, func):
##thrd               win.Connect(-1, -1, EVT_RESULT_ID, func)
##thrd           EVT_RESULT(self.self.frame, self.OnResult)
##thrd   
##thrd           class   Worker(threading.Thread):
##thrd               def __init__ (self2):
##thrd                   threading.Thread.__init__(self2)
##thrd                   self2.start()
##thrd       
##thrd               def run(self2):
##thrd                   #print "split.WorkerThread - start"
##thrd   
##thrd                   self.lastHistImg = None
##thrd                   self.recalcHist__Done = 1
##thrd                   try:
##thrd                       while self.a>0 and \
##thrd                                 self.hist.__class__.__name__ != '_wxPyDeadObject':
##thrd                           wx.Usleep( workerInterval )
##thrd                           # print self.a
##thrd                           self.a += 1
##thrd                           # wx.PostEvent(self.op.GetParent(), NewReport(self.wb))
##thrd                           # print self.viewer.m_imgArr
##thrd                           # print self.lastHistImg
##thrd                           if (not self.viewer.m_imgArr is self.lastHistImg) and \
##thrd                                    self.recalcHist__Done:
##thrd   #                          if (not self.viewer.m_imgArr == self.lastHistImg) and \
##thrd   #                                 self.recalcHist__Done:
##thrd                               self.lastHistImg = self.viewer.m_imgArr
##thrd                               #time print self.a, "z=", self.z, 
##thrd                               self.recalcHist()
##thrd                               #time print "done."
##thrd                   except: # The C++ part of the GLViewer object has been deleted
##thrd                       pass
##thrd                   #print "split.WorkerThread exited"
##thrd   

##thrd           if self.hist_arr is None:
##thrd            self.a = 1
##thrd               self.aa = Worker()

        #still __init__

    def onClose(self, ev=None):
        #print "debug: splitND::onClose"
        try:
            del self.data
            del self.img
        except:
            if PriConfig.raiseEventHandlerExceptions:
                raise
            else:
                import traceback, sys
                print >>sys.stderr, "  ### ### cought exception for debugging:  #### " 
                traceback.print_exc()
                print >>sys.stderr, "  ### ### cought exception for debugging:  #### " 
            
        from .usefulX import viewers
        try:
            viewers[ self.id ] = None
        except:
            if PriConfig.raiseEventHandlerExceptions:
                raise
            else:
                import traceback, sys
                print >>sys.stderr, "  ### ### cought exception for debugging:  #### " 
                traceback.print_exc()
                print >>sys.stderr, "  ### ### cought exception for debugging:  #### " 
        if ev:
            ev.GetEventObject().Destroy()
        #20070808self.frame.Destroy()
        # import gc
        # wx.CallAfter( gc.collect )
        
    #FIXME: size=(width+20,height+50+100)) # 20070627 MSW: was height+120
    def makeFrame(self, size): #20110902, title):
        """
        create frame
        if data has `meta.filename` attribute, append "<filename>" 
           (using os.path.basename) to given title
        """
        self.downSizeToFitWindow=False
        ### size = (400,400)
        if size is None:
            height,width = self.data.shape[-2:] #20051128 self.img.shape
            if height//2 == (width-1):  ## real_fft2d
                width = (width-1)*2
            if width>600 or height>600:
                width=height=600
                self.downSizeToFitWindow=True
#22             if self.nz > 1 and width<250: #HACK: minimum width to see z-slider
#22                 width = 250
        elif type(size) == int:
            width,height = size,size
        else:
            width,height = size
            
        '''
        #20110902
        try:
            fn = self.data.meta.filename
        except AttributeError:
            if title is None:
                title=''
        else:
            import os.path
            fn = os.path.basename(fn)
            if title:
                title = "%s <%s>" % (title,fn)
            else:
                title = "<%s>" % (fn,)
                
        # if hasattr(self.data, 'Mrc'): # was a HACK: and (len(title)<1 or title[-1]!='>'):
        #     ttt = "<%s>" % self.data.Mr.filename  # HACK, this should be done without the 'not in' check 
        #     if ttt not in title:
        #         if title !='':
        #             title += " "
        #         title += ttt
        
        title2 = "%d) %s" %(self.id, title)
        '''
        title2 = "" #20110902 -- self.setFrameTitle(title)  needs to be called later
        frame = wx.Frame(None, -1, title2, size=(width+20,height+50+100)) # 20070627 MSW: was height+120
        #20110902 from .usefulX import shellMessage
        #20110902 shellMessage("# window: %s\n"% title2)
        #20110902 self.title  = title
        #20110902 self.title2 = title2
        return frame


    '''
    def copyDataIfUnsupportedType(self, data):
        self.dataIsCplx = False

        if     data.type() == na.Int32:
            print "** split-viewer: converted Int32 to Int16"
            data = data.astype(na.Int16)
        elif   data.type() == na.UInt32:
            print "** split-viewer: converted UInt32 to UInt16"
            data = data.astype(na.UInt16)
        elif   data.type() == na.Float64:
            print "** split-viewer: converted Float64 to Float32"
            data = data.astype(na.Float32)
        elif data.type() == na.Complex64:
            print "** split-viewer: converted Complex64 to Complex32"
            self.dataCplx = data.astype(na.Complex32)
            self.dataIsCplx = True
            self.dataCplxShowAbsNotPhase = True
            data = na.abs(self.dataCplx)

        elif data.type() == na.Complex32:
            print "** split-viewer: complex - used abs()"
            self.dataCplx = data
            self.dataIsCplx = True
            self.dataCplxShowAbsNotPhase = True
            data = na.abs(self.dataCplx)

        self.data = data
    '''
        
    def OnAutoHistScale(self, event=77777):
        from .useful import mmms
        mi,ma,me,ss = mmms( self.img )
        self.hist.autoFit(amin=mi, amax=ma)
    def OnViewFFT(self, event=77777):
        from . import fftfuncs as F
        if self.data.dtype.type in (N.complex64, N.complex128):
            f = F.fft2d(self.data)
            #f[ ... , 0,0] = 0. # force DC to zero to ease scaling ...
            run(f, title='cFFT2d of %d'%self.id)
        else:
            f = F.rfft2d(self.data)
            f[ ... , 0,0] = 0. # force DC to zero to ease scaling ...
            run(f, title='rFFT2d of %d'%self.id)
#   def On88(self, event):
#       import fftfuncs as F
# #         if self.dataIsCplx:
# #             f = F.fft2d(self.dataCplx)
# #             run(f, title='cFFT2d of %d'%self.id)
# #         else:
#       f = F.irfft2d(self.data)
#       run(f, title='irFFT2d of %d'%self.id)
    def OnViewFFTInv(self, event=77777):
        from . import fftfuncs as F
        if self.data.dtype.type in (N.complex64, N.complex128):
            f = F.irfft2d(self.data)
        else:
            wx.Bell()
            return
        #    f = F.irfft2d(self.data)
        run(f, title='irFFT2d of %d'%self.id)
                
    def OnViewCplxAsPhase_or_Abs(self, event=77777):
        if not self.data.dtype.type in (N.complex64, N.complex128):
            wx.Bell()
            return
        self.viewer.m_viewComplexAsAbsNotPhase ^= True # toggle
        #import useful as U
        #self.data = U.phase(self.dataCplx)
        self.helpNewData()
    def OnViewFlipXZ(self, event=77777):
        from . import fftfuncs as F
        if self.data.dtype.type in (N.complex64, N.complex128):
            print "TODO: cplx "
        run(F.getXZview(self.data, zaxis=0), title='X-Z of %d'%self.id)
        from .usefulX import vHistSettingsCopy
        vHistSettingsCopy(self.id,-1)
    def OnViewFlipYZ(self, event=77777):
        from . import fftfuncs as F
        if self.data.dtype.type in (N.complex64, N.complex128):
            print "TODO: cplx "
        run(F.getYZview(self.data, zaxis=0), title='Y-Z of %d'%self.id)
        from .usefulX import vHistSettingsCopy
        vHistSettingsCopy(self.id,-1)

    def OnShowPopupTransient(self, evt=77777):
        try:
            print self.win
        except:
            print 'pass'
            pass
        self.win = TestTransientPopup(self.frame, wx.SIMPLE_BORDER)

        # Show the popup right below or above the button
        # depending on available screen space...
        #btn = evt.GetEventObject()
        #pos = btn.ClientToScreen( (0,0) )
        #sz =  btn.GetSize()
        #win.Position(pos, (0, sz.height))
        self.win.Position(self.frame.GetPosition(), (0,0) )
        
        self.win.Popup()

    def onGammaShortcut(self):
        from . import usefulX as Y
        Y.vGammaGUI(self.viewer)
        
    def helpNewData(self, doAutoscale=True, setupHistArr=True):
        """doAutoscale gets ORed with self.autoHistEachSect == 2
        """
        #self.zshape= data.shape[:-2]
        #self.zndim = len(self.zshape)
        ###self.img  = data[ (0,)*self.zndim ]
        #self.zsec  = [0] * self.zndim
        #self.zlast = [-1]*self.zndim # remember - for checking if update needed
        self.img =  self.data[tuple(self.zsec)]
        if self.img.dtype.type in (N.complex64, N.complex128):
            if self.viewer.m_viewComplexAsAbsNotPhase:
                #BAD20060302 self.img = abs(N.asarray(self.img, N.Float32))# check if this does tempcopy
                # From: Todd Miller <jmiller@stsci.edu>
                # Subject: Re: [Numpy-discussion] numarray: need Float32 abs
                #              from array of type na.Complex64 or na.Complex32
                # Date: Thu, 02 Mar 2006 10:13:32 -0500

                # stores the abs() into the real component of the original array.
                #         img.real is a view not a copy.
                #error-for-read-only-arrays  na.abs(self.img, self.img.real)   
                # optional step which makes the complex img array
                #      a real valued array with complex storage.
                #self.img.imag = 0               
                # just forget that img is using complex storage.
                self.img = N.asarray(N.absolute(self.img), N.float32)
            else:
                #from .all import U
                #data = U.phase(self.m_imgArr.astype(N.float32)
                #not temp copy for type conversion:
                self.img =  N.arctan2(N.asarray(self.img.imag, N.float32),
                                      N.asarray(self.img.real, N.float32))
            
        
        self.viewer.setImage( self.img )
        #print "debug1:", self.mmms
        #CHECK
        if setupHistArr:
            self.setupHistArr()
        if not self.noHistUpdate: # used for debugging speed issues
            self.recalcHist(triggeredFromIdle=True)
        if doAutoscale or self.autoHistEachSect == 2:
            self.hist.autoFit(amin=self.mmms[0], amax=self.mmms[1])
            #h.setBraces(self.mmms[0], self.mmms[1])
            #h.fitXcontrast()
            #self.viewer.changeHistogramScaling(self.mmms[0],self.mmms[1])
        elif self.autoHistEachSect == 1:
            self.hist.setBraces(self.mmms[0], self.mmms[1])
        #print "debug2:", self.mmms

    def OnHistLog(self, ev=77777):
        self.hist.OnLog(ev)
    def OnEnterScale(self, ev=77777):
        self.hist.OnEnterScale(ev)

    def OnViewVTK(self, ev=77777):
        from . import usefulX as Y
        ####use tostring instead    self.m = Y.vtkMountain(self.img, "vtk of %d: %s" %  (self.id, self.title))
        #20060722 a = N.NumArray(shape=self.img.shape, type=self.img.type(), buffer=self.viewer.m_imgArrString)
        a = N.fromstring(self.viewer.m_imgArrString, self.img.dtype)
        a.shape = self.img.shape
        self.m = Y.vtkMountain(a, "vtk of %d: %s" %  (self.id, self.title))

##thrd       def OnResult(self, event):
##thrd           #if event.data is None:
##thrd           self.hist.setHist(self.recalcHist__a_h, self.recalcHist__mmms[0], self.recalcHist__mmms[1])
        
    def setupHistArr(self):
        self.hist_arr = None

        if self.img.dtype.type == N.uint8:
            self.hist_min, self.hist_max = 0, (1<<8)-1
        elif self.img.dtype.type ==  N.uint16:
            self.hist_min, self.hist_max = 0, (1<<16)-1
        elif self.img.dtype.type == N.int16:
            self.hist_min, self.hist_max = 0-(1<<15), (1<<15)-1

        if self.img.dtype.type in (N.uint8, N.int16, N.uint16):
            self.hist_range = self.hist_max - self.hist_min + 1
            self.hist_arr = N.zeros(shape=self.hist_range, dtype=N.int32)
            

    def OnIdle(self, event):
        if len(self.recalcHist_todo_Set):
            i = self.recalcHist_todo_Set.pop()
            self.recalcHist(triggeredFromIdle=True)


    def recalcHist(self, triggeredFromIdle):
        if not triggeredFromIdle:
            self.recalcHist_todo_Set.add(0)
            return

        #CHECK img = self.viewer.m_imgArr
        img = self.img
        from . import useful as U
        mmms = U.mmms( img )
        self.mmms = mmms
            #time import time
            #time x = time.clock()
            # print mmms

        if self.hist_arr is not None:
            #glSeb  import time
            #glSeb  x = time.clock()
            U.histogram(img, amin=self.hist_min, amax=self.hist_max, histArr=self.hist_arr)
            self.hist.setHist(self.hist_arr, self.hist_min, self.hist_max)
            #glSeb  print "ms: %.2f"% ((time.clock()-x)*1000.0)
            ## FIXME  setHist needs to NOT alloc xArray every time !!!
        else:
        
            #          self.viewer.m_imgChanged = True
            #          self.viewer.Refresh(False)
    
            #20040915(OverflowError: float too large to convert)            resolution = int(mmms[1]-mmms[0]+2)
            #20040915if resolution > 10000:
            #20040915   resolution = 10000
            #20040915elif resolution < 1000: #CHECK
            #20040915   resolution = 10000 # CHECK
            resolution = 10000
    
            a_h = U.histogram(img, resolution, mmms[0], mmms[1])

            #    self.hist.setHist(a_h, mmms[0], mmms[1])
            self.recalcHist__a_h = a_h
            self.recalcHist__Done = 1
            #time print "recalcHist ms: %.2f"% ((time.clock()-x)*1000.0)
            if wx.Thread_IsMain():
                self.hist.setHist(self.recalcHist__a_h,
                                  self.mmms[0],
                                  self.mmms[1])
            else:
                wx.PostEvent(self.frame, self.__class__.ResultEvent(None))









class TestTransientPopup(wx.PopupTransientWindow):
    """Adds a bit of text and mouse movement to the wxPopupWindow"""
    def __init__(self, parent, style):
        wx.PopupTransientWindow.__init__(self, parent, style)
        panel = wx.Panel(self, -1)
        panel.SetBackgroundColour("#FFB6C1")
        st = wx.StaticText(panel, -1,
                          "wxPopupTransientWindow is a\n"
                          "wxPopupWindow which disappears\n"
                          "automatically when the user\n"
                          "clicks the mouse outside it or if it\n"
                          "(or its first child) loses focus in \n"
                          "any other way."
                          ,
                          pos=(10,10))
        sz = st.GetBestSize()
        panel.SetSize( (sz.width+20, sz.height+20) )
        self.SetSize(panel.GetSize())

        #wx.EVT_KEY_DOWN(self, self.OnKeyDown)
        #wx.EVT_KEY_UP(self, self.OnKeyUp)
        #wx.EVT_CHAR(self, self.OnChar)
        #self.SetFocus()

    def ProcessLeftDown(self, evt):
        #print "ProcessLeftDown"
        #self.Dismiss()
        return False

    #def OnDismiss(self):
    #   print "OnDismiss"

    def OnKeyDown(self, evt):
        print "OnKeyDown"
        #self.Dismiss()

    def OnKeyUp(self, evt):
        print "OnKeyUp"
        
    def OnChar(self, evt):
        print "OnKeyChar"
