"""
provides the bitmap OpenGL panel for Priithon's ND 2d-section-viewer 

common base class for single-color and multi-color version
"""
from __future__ import absolute_import
__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"


### NOTES 2008-July-04:
###
### rename m_init to m_glInited  # 20100122 - done.
### fix wheel for 2d images
### 
### revive idea that an image (texture) is handled within a m_moreGlLists (for multi-color viewer)
###
### indices in  m_moreGlLists[idx] are always growing - remove just sets m_moreGlLists[idx] to None

import wx
from wx import glcanvas
#from wxPython import glcanvas
from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL

import numpy as N
import traceback
from . import PriConfig

bugXiGraphics = 0

Menu_Zoom2x      = wx.NewId()
Menu_ZoomCenter  = wx.NewId()
Menu_Zoom_5x     = wx.NewId()
Menu_ZoomReset   = wx.NewId()
Menu_Zoom1       = wx.NewId()
Menu_ZoomOut     = wx.NewId()
Menu_ZoomIn      = wx.NewId()
Menu_Color       = wx.NewId()
Menu_Reload       = wx.NewId()
Menu_chgOrig     = wx.NewId()
Menu_Save = wx.NewId()
Menu_SaveScrShot = wx.NewId()
Menu_SaveClipboard = wx.NewId()
Menu_Assign = wx.NewId()
Menu_noGfx = wx.NewId()
Menu_aspectRatio = wx.NewId()
Menu_rotate = wx.NewId()
Menu_grid        = wx.NewId()
Menu_ColMap = [wx.NewId() for i in range(8)]

'''
screen_max_x, screen_max_y = wx.DisplaySize()
screen_max_x, screen_max_y = screen_max_x-1, screen_max_y-1
'''
class GLViewerCommon(glcanvas.GLCanvas):
    def __init__(self, parent, size=wx.DefaultSize):
        glcanvas.GLCanvas.__init__(self, parent, -1, size=size, style=wx.WANTS_CHARS)
        # wxWANTS_CHARS to get arrow keys on Windows

        self.error = None
        self.m_doViewportChange = True
    
        # NEW 20080701:  in new coord system, integer pixel coord go through the center of pixel
        self.x00 = -.5 # 0
        self.y00 = -.5 # 0
        self.m_x0=None #20070921 - call center() in OnPaint -- self.x00
        self.m_y0=None #20070921 - call center() in OnPaint -- self.y00
        self.m_scale=1
        self.m_aspectRatio = 1.
        self.m_rot=0.
        self.m_zoomChanged = True # // trigger a first placing of image
        self.keepCentered = True

        #20080722 self.m_pixelGrid_Idx = None
        self.m_pixelGrid_state = 0 # 0-off, 1-everyPixel, 2- every 10 pixels

        self.m_glInited   = False
        try:
            self.context = glcanvas.GLContext(self)
        except TypeError: # wx < 2.9.1
            # isRGB__unused= 0
            # glcanvas.GLContext(isRGB__unused, self)
            self.context = self .GetContext() 

        self.m_moreGlLists = []
        self.m_moreGlLists_enabled = []
        self.m_moreMaster_enabled = True
        self.m_moreGlLists_dict = {} # map 'name' to list of idx in m_moreGlLists
        # a given idx can be part of multiple 'name' entries
        # a given name entry can contain a given idx only once
        # a name that is a tuple, has a special meaning: if name == zSecTuple - 
        #               auto en-/dis-able gfxs in splitNDcommon::OnZZSlider (ref. zlast)
        #               UNLESS gfx idx is in self.m_moreGlLists_nameBlacklist
        self.m_moreGlLists_nameBlacklist = set()
        self.m_moreGlLists_NamedIdx = {} # map str to int or None -- this is helpful for reusing Idx for "changing" gllists
                                         # if name (type str) wasn't used before, it defaults to None (20080722)

        self.m_moreGlListReuseIdx=None
        self.m_wheelFactor = 2 ** (1/3.) #1.189207115002721 # >>> 2 ** (1./4)  # 2
        self.mouse_last_x, self.mouse_last_y = 0,0 # in case mouseIsDown happens without preceeding mouseDown

        #20080707 doOnXXXX event handler are now lists of functions
        #                x,y are (corrected, float value) pixel position
        #                ev is the wx onMouseEvent obj -- use ev.GetEventObject() to get to the viewer object
        #20080707-unused self.doOnFrameChange = [] # no args
        self.doOnMouse       = [] # (x,y, ev)
        self.doOnLDClick     = [] # (x,y, ev)
        self.doOnLDown       = [] # (x,y, ev)
        self.doOnPanZoom     = [] # ( ???? )


        wx.EVT_ERASE_BACKGROUND(self, self.OnEraseBackground)
        #20080707-unused wx.EVT_MOVE(parent, self.OnMove) # CHECK
        wx.EVT_SIZE(self, self.OnSize)
        self.OnWheel = self.OnWheel_zoom  # this variable points to the current onWheel function  (val might change to OnWheel_scroll)
        wx.EVT_MOUSEWHEEL(self, self.OnWheel)
        wx.EVT_MOUSE_EVENTS(self, self.OnMouse)
        #self.Bind(wx.EVT_SIZE, self.OnSize)

        if wx.VERSION >= (2,9):  # API change with wx 2.9 --
            # http://wxpython-users.1045709.n5.nabble.com/GLCanvas-and-GLContext-td2651535.html
            self.set_current = lambda : self.SetCurrent(self.context)
        else:
            self.set_current = self.SetCurrent

        if wx.version()=='2.9.3.1 msw (classic)':
            def ReleaseMouse_SebMSW293bugfix():
                try:
                    super(GLViewerCommon, self).ReleaseMouse()
                except wx.PyAssertionError: 
                    pass
            self.ReleaseMouse = ReleaseMouse_SebMSW293bugfix


    def setPixelGrid(self, ev=None):
        if   self.m_pixelGrid_state == 0:  # old state == 0 == 'off'             -> new state=1
            self.m_pixelGrid_state = 1
            self.drawPixelGrid(1, 1)
        elif self.m_pixelGrid_state == 1:  # old state == 1 == 'every pixel'     -> new state=2
            self.m_pixelGrid_state = 2
            self.drawPixelGrid(10, 10)
        elif self.m_pixelGrid_state == 2:  # old state == 2 == 'every 10 pixels'  -> new state=0
            self.m_pixelGrid_state = 0

            #self.newGLListRemove( self.m_pixelGrid_Idx )
            self.newGLListEnable( 'm_pixelGrid_Idx', False )

    def drawPixelGrid(self, spacingY, spacingX, color=(1,0,0), width=1):
        self.newGLListNow(idx='m_pixelGrid_Idx')
        glLineWidth(width)
        glColor(*color)
        glTranslate(-.5,-.5 ,0)  # 20080701:  in new coord system, integer pixel coord go through the center of pixel

        glBegin(GL_LINES)
        ny = self.pic_ny
        nx = self.pic_nx
        if self.m_originLeftBottom == 8:
            nx = (nx-1)*2
        for y in N.arange(0,ny+.5, spacingY):
            glVertex2d(0, y)
            glVertex2d(nx, y)
        for x in N.arange(0,nx+.5, spacingX):
            glVertex2d(x, 0)
            glVertex2d(x, ny)
        glEnd()

        glTranslate(.5,.5 ,0)  # 20080701:  in new coord system, integer pixel coord go through the center of pixel            
        self.newGLListDone(enable=True, refreshNow=True)
        





    def newGLListNow(self, name=None, idx=None) : # , i):
        """
        call this immediately before you call a bunch of gl-calls
        issue newGLListDone() when done
        OR newGLListAbort() when there is problem and
            the glist should get cleared

        create new or append to dict entry 'name' when done
            if name is a list (! not tuple !) EACH list-items is used
            a tuple is interpreted as "z-sect-tuple" and means that this gllist 
                gets automacically en-/dis-abled with z-slider entering/leaving that section
                (see on splitNDcommon::OnZZSlider)
        if idx is not None:  reuse and overwrite existing gllist
        if idx is of type str: on first use, same as None; but on subsequent uses, reuse and overwrite 
        """
        self.m_moreGlListReuseIdx = idx
        self.set_current() # 2012 self.SetCurrent(self.context) # 20111103
        if isinstance(idx, basestring):
            idx = self.m_moreGlLists_NamedIdx.get(idx) # Never raises an exception if k is not in the map, instead it returns x. x is optional; when x is not provided and k is not in the map, None is returned. 
            if idx is None:
                self.curGLLIST = glGenLists( 1 )
            else:
                try:
                    self.curGLLIST = self.m_moreGlLists[idx]
                except IndexError: ## vgRemoveAll might have been called
                    self.curGLLIST = glGenLists( 1 )
                    del self.m_moreGlLists_NamedIdx[self.m_moreGlListReuseIdx] # will get reset in newGLListDone()
                else:
                    if self.curGLLIST is None:   # glList was deleted Y.vgRemove
                        self.curGLLIST = glGenLists( 1 )

        elif idx is None or self.m_moreGlLists[idx] is None:
            self.curGLLIST = glGenLists( 1 )
        else:
            self.curGLLIST = self.m_moreGlLists[idx]

        self.curGLLISTname  = name
        glNewList( self.curGLLIST, GL_COMPILE )

    def newGLListAbort(self):
        glEndList()
        glDeleteLists(self.curGLLIST, 1)
        if isinstance(self.m_moreGlListReuseIdx, basestring):
            try:
                del self.m_moreGlLists_NamedIdx[self.m_moreGlListReuseIdx] # CHECK
            except KeyError:
                pass # was not in dict yet
        self.m_moreGlListReuseIdx = None

    def newGLListDone(self, enable=True, refreshNow=True):
        glEndList()
        if isinstance(self.m_moreGlListReuseIdx, basestring):
            idx = self.m_moreGlLists_NamedIdx.get(self.m_moreGlListReuseIdx)
        else:
            idx = self.m_moreGlListReuseIdx

        if idx is not None:
            self.m_moreGlLists[idx] = self.curGLLIST # left side might have been None
            self.m_moreGlLists_enabled[idx] = enable
        else:
            idx = len(self.m_moreGlLists)
            self.m_moreGlLists.append( self.curGLLIST )
            self.m_moreGlLists_enabled.append( enable )
        
        self.newGLListNameAdd(idx, self.curGLLISTname)

        # remember named idx for future re-use
        if isinstance(self.m_moreGlListReuseIdx, basestring):
            self.m_moreGlLists_NamedIdx[self.m_moreGlListReuseIdx] = idx
        self.m_moreGlListReuseIdx = None

        if refreshNow:
            self.Refresh(0)
        return idx

    def newGLListNameAdd(self, idx, name):
        if type(name) != list:
            name = [ name ]

        # make sure cur idx is in each name-list; create new name-list or append to existing, as needed 
        for aName in name:
            if aName is not None:
                try:
                    l = self.m_moreGlLists_dict[aName]
                    try:
                        l.index(idx)  # don't do anything if aName is already in
                    except ValueError:
                        l.append(idx)
                except KeyError:
                    self.m_moreGlLists_dict[aName] = [idx]
    def newGLListNameRemove(self, idx, name):
        if type(name) != list:
            name = [ name ]

        # remove idx from list given by each name
        for aName in name:
            if aName is not None:
                try:
                    l = self.m_moreGlLists_dict[aName]
                    try:
                        l.remove( idx )
                    except ValueError:
                        # don't do anything if idx was not part of aName
                        pass
                except KeyError:
                    pass

    def newGLListRemove(self, idx, refreshNow=True):
        """
        instead of 'del' just set entry to None
        this is to prevent, shifting of all higher idx
        20090107: but do 'del' for last entry - no trailing Nones

        self.m_moreGlLists_dict is cleaned properly
        """
        #20070712 changed! not 'del' - instead set entry to None
        #20070712    ---- because decreasing all idx2 for idx2>idx is complex !!!
        #untrue note;  --- old:
        #untrue note; be careful: this WOULD change all indices (idx) of GLLists
        #untrue note; following idx
        #untrue note!!: if you can not accept that: you should call
        #untrue note!!:   newGLListEnable(idx, on=0)

#       if self.m_moreGlLists_texture[idx] is not None:
#           glDeleteTextures( self.m_moreGlLists_texture[idx] )
#           del self.m_moreGlLists_img[idx]

        if isinstance(idx, basestring):
            idx=self.m_moreGlLists_NamedIdx[idx]
        elif idx<0:
            idx += len(self.m_moreGlLists)

        if self.m_moreGlLists[idx]: # could be None - # Note: Zero is not a valid display-list index.
            glDeleteLists(self.m_moreGlLists[idx], 1)
        #20070712 del self.m_moreGlLists[idx]
        #20070712 del self.m_moreGlLists_enabled[idx]
        if idx == len(self.m_moreGlLists)-1: # 20090107
            del self.m_moreGlLists[idx]
            del self.m_moreGlLists_enabled[idx]
        else:
            self.m_moreGlLists[idx] = None
            self.m_moreGlLists_enabled[idx] = None
        self.m_moreGlLists_nameBlacklist.discard(idx)

        #remove idx from 'name' dict entry
        #   remove respective dict-name if it gets empty
        _postposeDelList = [] # to prevent this error:dictionary changed size during iteration
        for name,idxList in self.m_moreGlLists_dict.iteritems():
            try:
                idxList.remove(idx)
                if not len(idxList):
                    _postposeDelList.append(name)
            except ValueError:
                pass
        for name in _postposeDelList:
            del self.m_moreGlLists_dict[name]

        if refreshNow:
            self.Refresh(0)

    def newGLListEnable(self, idx, on=True, refreshNow=True):
        """
        ignore moreGlList items that are None !
        """
        if isinstance(idx, basestring):
            idx=self.m_moreGlLists_NamedIdx[idx]
        if self.m_moreGlLists_enabled[idx] is not None:
            self.m_moreGlLists_enabled[idx] = on
        if refreshNow:
            self.Refresh(0)

    def newGLListEnableByName(self, name, on=True, skipBlacklisted=False, refreshNow=True):
        """
        "turn on/off" all gfx whose idx is in name-dict
        if skipBlacklisted: IGNORE idx if contained in moreGlLists_nameBlacklist
        ignore moreGlList items that are None !
        """
        for idx in self.m_moreGlLists_dict[name]:
            if self.m_moreGlLists_enabled[idx] is not None and\
                    (not skipBlacklisted or idx not in self.m_moreGlLists_nameBlacklist):
                self.m_moreGlLists_enabled[idx] = on
        if refreshNow:
            self.Refresh(0)

    def newGLListRemoveByName(self, name, refreshNow=True):
        for idx in self.m_moreGlLists_dict[name]:
            if self.m_moreGlLists[idx]:
                glDeleteLists(self.m_moreGlLists[idx], 1)
            # refer to comment in newGLListRemove() !!!
            self.m_moreGlLists[idx]  = None
            self.m_moreGlLists_enabled[idx]  = None
        del self.m_moreGlLists_dict[name]

        # clean up other name entries in dict
        for name,idxList in self.m_moreGlLists_dict.items():
            for i in xrange(len(idxList)-1,-1,-1):
                if self.m_moreGlLists[idxList[i]] is None:
                    del idxList[i]
            if not len(idxList):
                del self.m_moreGlLists_dict[name]


        #20090505: remove trailing None's
        for idx in xrange(len(self.m_moreGlLists)-1, -1, -1):
            if self.m_moreGlLists[idx] is None:
                del self.m_moreGlLists[idx]
                del self.m_moreGlLists_enabled[idx]
            else:
                break

        #20100505: remove named indices if they got invalid
        for name,idx in self.m_moreGlLists_NamedIdx.items(): 
            # py3k: is items() still allowing `del` - no copy made anymore
            if idx>=len(self.m_moreGlLists):
                del self.m_moreGlLists_NamedIdx[name]

        if refreshNow:
            self.Refresh(0)

    def newGLListRemoveAll(self, refreshNow=True):
        """
        this really removes all GLList stuff
        idx values will restart at 0
        here nothing gets "only" set to None
        """
        for li in self.m_moreGlLists:
            if li:  # Note: Zero is not a valid display-list index.
                glDeleteLists(li, 1)
        self.m_moreGlLists = []
        self.m_moreGlLists_enabled = []
        #self.m_moreMaster_enabled = 1
        self.m_moreGlLists_dict.clear()
        self.m_moreGlLists_nameBlacklist.clear()
        self.m_moreGlLists_NamedIdx.clear()

        if refreshNow:
            self.Refresh(0)
        




    def OnNoGfx(self, evt):
        #fails on windows:
        if wx.Platform == '__WXMSW__': ### HACK check LINUX GTK WIN MSW
            menuid  = self.m_menu.FindItem("hide all gfx")
            self.m_menu.FindItemById(menuid).Check( not evt.IsChecked() )
            self.m_moreMaster_enabled ^= 1
        else:
            self.m_moreMaster_enabled = not evt.IsChecked()

        self.Refresh(0)

    def OnChgNoGfx(self):
        self.m_moreMaster_enabled ^= 1
        menuid  = self.m_menu.FindItem("hide all gfx")
        self.m_menu.FindItemById(menuid).Check(not self.m_moreMaster_enabled)
        self.Refresh(0)

    def setAspectRatio(self, y_over_x, refreshNow=1):
        """
        strech images in y direction
        use negative value to mirror
        """
        
        self.m_aspectRatio=y_over_x
        
        self.m_zoomChanged=True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        if refreshNow:
            self.Refresh()

    def setRotation(self, angle=90, refreshNow=1):
        """rotate everything by angle in degrees
        """
        
        self.m_rot = angle
        
        self.m_zoomChanged=True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        if refreshNow:
            self.Refresh()

    def setOriginLeftBottom(self, olb):
        hasToReloadImgArr = not (self.m_originLeftBottom < 2 and olb < 2)

        if (olb == 0 and self.m_originLeftBottom in (1,7,8)  or
            olb in (1,7,8) and self.m_originLeftBottom == 0):
            # switch from origin at left bottom to left top  OR
            # switch from origin at left top to left bottom

            self.m_y0 += self.pic_ny*self.m_scale* self.m_aspectRatio
            self.m_aspectRatio *= -1
            self.m_zoomChanged = True
            #CHECK from .usefulX import _callAllEventHandlers
            #CHECK _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")

        hasChanged = self.m_originLeftBottom != olb
        self.m_originLeftBottom = olb
        if hasToReloadImgArr:
            self.m_imgToDo = self.m_imgArr
            self.InitTex()

        self.m_imgChanged=True  #20110902 apparently the displayed graphics gets "often" corrupted
        if hasChanged:            
            try:
                self.my_spv.setFrameTitle(self.my_spv.title, appendFilename=False)
            except AttributeError:
                pass
        self.Refresh(0)


    def center(self, refreshNow=True):
        self.keepCentered = True

        ws = N.array([self.m_w, self.m_h])
        nx, ny = self.pic_nx, self.pic_ny
        if self.m_originLeftBottom == 8:
            nx = (self.pic_nx-1) * 2
        ps = N.array([nx, ny])
        s  = self.m_scale,self.m_scale*self.m_aspectRatio
        self.m_x0, self.m_y0 = (ws-(ps-1)*s) // 2
        if self.m_originLeftBottom in (7,8):
            self.m_x0= self.m_x0 + self.m_scale*nx/2
            self.m_y0= self.m_y0 + self.m_scale*self.m_aspectRatio*ny/2
        self.m_zoomChanged = True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        if refreshNow:
            self.Refresh(0)

    def flipY(self, refreshNow=True): # ,x):
        self.m_aspectRatio *= -1
        self.m_y0 -= (self.pic_ny-1) * self.m_scale * self.m_aspectRatio
        #if x:
        #else:
        #    self.m_y0 -= self.pic_ny * self.m_scale * self.m_aspectRatio
        
        self.m_zoomChanged=True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        if refreshNow:
            self.Refresh(0)
        
        
    def zoom(self, zoomfactor=None, cyx=None, absolute=True, refreshNow=True):
        """
        set new zoom factor to zoomfactor
        if absolute is False
           adjust current zoom factor to
              "current"*zoomfactor
        if zoomfactor is None:
            zoomfactor stays unchanged

        if cyx is None:
            image center stays center
        otherwise, image will get "re-centered" to cyx beeing the new center
        """
        if zoomfactor is not None:
            if absolute:
                fac = zoomfactor / self.m_scale
            else:
                fac = zoomfactor
            self.m_scale *= fac

        w2 = self.m_w/2
        h2 = self.m_h/2
        if cyx is None:
            self.m_x0 = w2 - (w2-self.m_x0)*fac
            self.m_y0 = h2 - (h2-self.m_y0)*fac
        else:
            cy,cx = cyx
            self.m_x0 = w2 - cx*self.m_scale
            self.m_y0 = h2 - cy*(self.m_scale*self.m_aspectRatio)
            
        self.m_zoomChanged = True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        if refreshNow:
            self.Refresh(0)

    def doReset(self, ev=None, refreshNow=True):
        self.keepCentered = False

        self.m_x0=self.x00
        self.m_y0=self.y00
        self.m_scale=1.
        self.m_rot=0.
        self.m_aspectRatio = 1.
        self.m_zoomChanged = True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        if refreshNow:
            self.Refresh(0)



    def OnChgOrig(self, event=None):
        o = self.m_originLeftBottom
        if o == 1:
            self.setOriginLeftBottom(0)
        elif o == 0:
            self.setOriginLeftBottom(7)
        elif o == 7:
            self.setOriginLeftBottom(8)
        elif o == 8:
            self.setOriginLeftBottom(1)
        else:
            raise RuntimeError, "should never get here ---- FixMe: OnChgOrig"

    def OnCenter(self, event=None): # was:On30
        self.center()
    def OnZoomOut(self, event=77777): # was:On31
        fac = 1./1.189207115002721 # >>> 2 ** (1./4)
        self.zoom(fac, absolute=False)        
    def OnZoomIn(self, event=77777): # was:On32
        fac = 1.189207115002721 # >>> 2 ** (1./4)
        self.zoom(fac, absolute=False)

#      def On41(self, event):
#          self.doShift(- self.m_scale , 0)
#      def On42(self, event):
#          self.doShift(+ self.m_scale , 0)
#      def On43(self, event):
#          self.doShift(0,  + self.m_scale)
#      def On44(self, event):
#          self.doShift(0,  - self.m_scale)

    def quaterShiftOffsetLeft(self):
        n= self.pic_nx / 4
        if self.m_originLeftBottom == 8:
            n= (n-1) * 2
        self.doShift(int(- self.m_scale*n) , 0)
    def quaterShiftOffsetRight(self):
        n= self.pic_nx / 4
        if self.m_originLeftBottom == 8:
            n= (n-1) * 2
        self.doShift(int(+ self.m_scale*n) , 0)
    def quaterShiftOffsetUp(self):
        n= self.pic_ny / 4
        self.doShift(0,  int(+ self.m_scale*self.m_aspectRatio*n))
    def quaterShiftOffsetDown(self):
        n= self.pic_ny / 4
        self.doShift(0,  int(- self.m_scale*self.m_aspectRatio*n))

    def doShift(self, dx,dy):
        """
        shift view offset by dx,dy pixels
        if dx or dy is a float, it is multiplied by the viewer's 
           width or height respectively
        keepCentered is set to False
        """
        self.keepCentered = False

        if isinstance(dx, float):
            dx = int(dx * self.m_w)
        if isinstance(dy, float):
            dy = int(dy * self.m_h)

        self.m_x0 += dx
        self.m_y0 += dy
        
        self.m_zoomChanged = True
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        self.Refresh(0)

    def OnMouse(self, ev):
        from .usefulX import _callAllEventHandlers
        self._onMouseEvt = ev  # be careful - only use INSIDE a handler function that gets call from here
        if self.m_x0 is None:
            return # before first OnPaint call

        self.set_current() # 2012 self.SetCurrent(self.context) # 20111103
        x,y = ev.GetX(),  self.m_h-ev.GetY()
        xEff_float, yEff_float= gluUnProject(x,y,0)[:2]

        # 20080701:  in new coord system, integer pixel coord go through the center of pixel

        #20080707-alwaysCall_DoOnMouse xyEffInside = False
        nx = self.pic_nx
        ny = self.pic_ny
        #20080707 xyEffVal = 0

        import sys
        if sys.platform not in ('win32', 'darwin') and ev.Entering():
            self.SetFocus()

        #20100325 if self.m_originLeftBottom == 0:
        #20100325     yEff_float = ny-1 - yEff_float
        #elif self.m_originLeftBottom == 1:
        #  pass

        midButt = ev.MiddleDown() or (ev.LeftDown() and ev.AltDown())
        midIsButt = ev.MiddleIsDown() or (ev.LeftIsDown() and ev.AltDown())
        rightButt = ev.RightDown() or (ev.LeftDown() and ev.ControlDown())
        
        # TODO CHECK 
        # Any application which captures the mouse in the beginning of some
        # operation must handle wxMouseCaptureLostEvent and cancel this
        # operation when it receives the event.
        # The event handler must not recapture mouse. 
        if self.HasCapture():
            if not (midIsButt or ev.LeftIsDown()):
                self.ReleaseMouse()
        else:
            if midButt or ev.LeftDown():
                self.CaptureMouse()

        #20070713 if ev.Leaving():
        #20070713     ## leaving trigger  event - bug !!
        #20070713     return

        if midButt:
            #20100125 print "# debug: save self.mouse_last_x, self.mouse_last_y", x,y
            self.mouse_last_x, self.mouse_last_y = x,y
        elif midIsButt: #ev.Dragging()
            self.keepCentered = False
            if ev.ShiftDown() or ev.ControlDown():
                #dx = x-self.mouse_last_x
                dy = y-self.mouse_last_y

                fac = 1.05 ** (dy)
                self.m_scale *= fac
                w2 = self.m_w/2
                h2 = self.m_h/2
                self.m_x0 = w2 - (w2-self.m_x0)*fac
                self.m_y0 = h2 - (h2-self.m_y0)*fac
                self.m_zoomChanged = True

            else:
                self.m_x0 += (x-self.mouse_last_x) #/ self.sx
                self.m_y0 += (y-self.mouse_last_y) #/ self.sy
            self.m_zoomChanged = True
            #20100125 print "# debug:222222self.mouse_last_x, self.mouse_last_y", self.mouse_last_x, self.mouse_last_y, x,y
            
            '''
            warp=False
            xMax,yMax = self.ScreenToClientXY(screen_max_x, screen_max_y)
            x0,  y0   = self.ScreenToClientXY(0,0)
            print x,y
            if x<=x0:
                x=xMax
                warp=True
            elif x>=xMax:
                x=x0
                warp=True
            if y<=y0:
                y=yMax
                warp=True
            elif y>=yMax:
                y=y0
                warp=True
            if warp:
                print '--------------->',x,y
                self.WarpPointer( x,y )
            '''
            self.mouse_last_x, self.mouse_last_y = x,y
            self.Refresh(0)

        elif rightButt:
            #20060726 self.mousePos_remembered_x, self.mousePos_remembered_y = ev.GetPositionTuple()
            pt = ev.GetPosition()
            self.PopupMenu(self.m_menu, pt)
        elif ev.LeftDown():
            _callAllEventHandlers(self.doOnLDown, (xEff_float,yEff_float, ev), "doOnLDown")
                    
        elif ev.LeftDClick():
            _callAllEventHandlers(self.doOnLDClick, (xEff_float,yEff_float, ev), "doOnLDClick")
                    
            #print ":", x,y, "   ", x0,y0, s, "   ", xyEffInside, " : ", xEff, yEff
            
            #if xyEffInside:
            #    self.doDClick(xEff, yEff)
            #self.doOnLeftDClick(ev)

        #20080707-alwaysCall_DoOnMouse if xyEffInside:
        _callAllEventHandlers(self.doOnMouse, (xEff_float,yEff_float, ev), "doOnMouse")

        if self.m_zoomChanged:
            _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
        ev.Skip() # other things like EVT_MOUSEWHEEL are lost




    def OnEraseBackground(self, ev):
        pass # do nothing to prevent flicker !!

    #20080707-unused def OnMove(self, event):
    #20080707-unused     self.doOnFrameChange()
    #20080707-unused     event.Skip()

    def OnSize(self, event):
        self.m_w, self.m_h = self.GetSizeTuple() # self.GetClientSizeTuple()
        if self.m_w <=0 or self.m_h <=0:
            #print "GLViewer.OnSize self.m_w <=0 or self.m_h <=0", self.m_w, self.m_h
            return
        self.m_doViewportChange = True

        if self.keepCentered and self.m_x0 is not None:
            self.center()
        event.Skip()

    def OnWheel_zoom(self, evt):
        #delta = evt.GetWheelDelta()
        rot = evt.GetWheelRotation()      / 120. #HACK
        #linesPer = evt.GetLinesPerAction()
        #print "wheel:", delta, rot, linesPer
        if 1:#nz ==1:
            zoomSpeed = 1. # .25
            fac = self.m_wheelFactor ** (rot*zoomSpeed) # 1.189207115002721 # >>> 2 ** (1./4)
            self.m_scale *= fac
            w2 = self.m_w/2
            h2 = self.m_h/2
            self.m_x0 = w2 - (w2-self.m_x0)*fac
            self.m_y0 = h2 - (h2-self.m_y0)*fac
            self.m_zoomChanged = True
            from .usefulX import _callAllEventHandlers
            _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
            self.Refresh(0)
        #else:
        #    slider.SetValue()
        evt.Skip() #?

    #20080707 def doLDClick(self, x,y):
    #20080707     # print "doDLClick xy: --> %7.1f %7.1f" % (x,y)
    #20080707     pass
    #20080707 def doLDown(self, x,y):
    #20080707     # print "doLDown xy: --> %7.1f %7.1f" % (x,y)
    #20080707     pass

        
    def OnSaveClipboard(self, event=None):
        from . import usefulX as Y
        Y.vCopyToClipboard(self, clip=1)
        Y.shellMessage("### screenshot saved to clipboard\n")

    def OnSaveScreenShort(self, event=None):
        """always flipY"""
        from .all import U, FN, Y
        fn = FN(1, verbose=0)
        if not fn:
            return

        flipY=1
        U.saveImg(self.readGLviewport(clip=True, flipY=flipY, copy=1), fn)
        
        Y.shellMessage("### screenshot saved to '%s'\n"%(fn,))

    def OnAssign(self, event=None):
        from . import usefulX as Y
        ss = "<2d section shown>"

        for i in range(len(Y.viewers)):
            try:
                v = Y.viewers[i]
                if v.viewer is self:
                    ss = "Y.vd(%d)[%s]"%(i, ','.join(map(str,v.zsec)))
                    break
            except:
                pass

        Y.assignNdArrToVarname(self.m_imgArr, ss)

    def OnSave(self, event=None):
        from .all import Mrc, U, Y
        fn = Y.FN(1, verbose=0)
        if not fn:
            return
        if fn[-4:] in [ ".mrc",  ".dat" ]:
            Mrc.save(self.m_imgArr, fn)
        elif fn[-5:] in [ ".fits" ]:
            U.saveFits(self.m_imgArr, fn)
        else:
            U.saveImg8(self.m_imgArr, fn)

        Y.shellMessage("### section saved to '%s'\n"%fn)

    def OnRotate(self, evt):
        from . import usefulX as Y
        Y.vRotate(self)
    def OnAspectRatio(self, evt):
        ds = "nx/ny"
        if self.m_originLeftBottom == 8:
            ds = "(2*nx+1)/ny"
        a = wx.GetTextFromUser('''\
set image aspect ratio (y/x factor for display)
  (any python-expression is OK)
     nx,ny = width,height
     a     = current aspect ratio                             
                               ''',
                               "set image aspect ratio",
                               ds)
        if a=='':
            return
        import __main__
        loc = { 'nx': float(self.pic_nx),
                'ny': float(self.pic_ny),
                'a' : self.m_aspectRatio,
                }
        try:
            y_over_x = float( eval(a,__main__.__dict__, loc) )
        except:
            raise # this was from the time before we had guiExceptions, I guess...
            #             import sys
            #             e = sys.exc_info()
            #             wx.MessageBox("Error when evaluating %s: %s - %s" %\
                #                           (a, str(e[0]), str(e[1]) ),
            #                           "syntax(?) error",
            #                           style=wx.ICON_ERROR)
        else:
            self.setAspectRatio(y_over_x)

    def OnMenu(self, event):
        id = event.GetId()
        
        #          if id == Menu_ZoomCenter:
        #              x = self.mousePos_remembered_x
        #              y = self.mousePos_remembered_y
        
        #              w2 = self.m_w/2
        #              h2 = self.m_h/2
        #              self.m_x0 += (w2-x)*self.m_scale
        #              self.m_y0 += (h2-y)*self.m_scale
        #              self.m_zoomChanged = True

        if id == Menu_Zoom2x:
            fac = 2.
            self.m_scale *= fac
            w2 = self.m_w/2.
            h2 = self.m_h/2.
            self.m_x0 = w2 - (w2-self.m_x0)*fac
            self.m_y0 = h2 - (h2-self.m_y0)*fac
            self.m_zoomChanged = True
        elif id == Menu_Zoom_5x:
            fac = .5
            self.m_scale *= fac
            w2 = self.m_w/2.
            h2 = self.m_h/2.
            self.m_x0 = w2 - (w2-self.m_x0)*fac
            self.m_y0 = h2 - (h2-self.m_y0)*fac
            self.m_zoomChanged = True

        if self.m_zoomChanged:
            from .usefulX import _callAllEventHandlers
            _callAllEventHandlers(self.doOnPanZoom, (), "doOnPanZoom")
            self.Refresh(0)


           

    def readGLviewport(self, clip=False, flipY=True, copy=True):
        """returns array with r,g,b values from "what-you-see"
            shape(3, height, width)
            type=UInt8

            if clip: clip out the "green background"
            if copy == 0 returns non-contiguous array!!!

        """
        self.set_current() # 2012 self.SetCurrent(self.context) # 20111103
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        
        get_cm = glGetInteger(GL_MAP_COLOR)
        get_rs = glGetDoublev(GL_RED_SCALE)
        get_gs = glGetDoublev(GL_GREEN_SCALE)
        get_bs = glGetDoublev(GL_BLUE_SCALE)
            
        get_rb = glGetDoublev(GL_RED_BIAS)
        get_gb = glGetDoublev(GL_GREEN_BIAS)
        get_bb = glGetDoublev(GL_BLUE_BIAS)

        glPixelTransferi(GL_MAP_COLOR, False)

        glPixelTransferf(GL_RED_SCALE,   1)
        glPixelTransferf(GL_GREEN_SCALE, 1)
        glPixelTransferf(GL_BLUE_SCALE,  1)
            
        glPixelTransferf(GL_RED_BIAS,   0)
        glPixelTransferf(GL_GREEN_BIAS, 0)
        glPixelTransferf(GL_BLUE_BIAS,  0)

        b=glReadPixels(0,0, self.m_w, self.m_h,
                       GL_RGB,GL_UNSIGNED_BYTE)
        
        bb=N.ndarray(buffer=b, shape=(self.m_h,self.m_w,3),
                   dtype=N.uint8) #, aligned=1)

        cc = N.transpose(bb, (2,0,1))

        if clip:
            x0,y0, s,a = int(self.m_x0), int(self.m_y0),self.m_scale,self.m_aspectRatio
            if hasattr(self, "m_imgArr"):
                ny,nx = self.m_imgArr.shape
            else:
                ny,nx = self.m_imgList[0][2].shape
            nx,ny = int(nx*s +.5), int(ny*s*a + .5)
            x1,y1 = x0+ nx, y0+ny
            if a<0:
                y0,y1=y1,y0
                y1+=1 # fix wrong .5 rounding  for ny

            x0 = N.clip(x0, 0, self.m_w)
            x1 = N.clip(x1, 0, self.m_w)
            y0 = N.clip(y0, 0, self.m_h)
            y1 = N.clip(y1, 0, self.m_h)
            #20110902 unused  nx,ny = x1-x0, y1-y0

            cc=cc[:,y0:y1,x0:x1]
        #else:
        #    y0,x0 = 0,0
        #    ny,nx = y1,x1 = self.m_h, self.m_w

        if flipY:
            cc = cc[:,::-1] # flip y
            
        if copy:
            cc = cc.copy()

        glPixelTransferi(GL_MAP_COLOR, get_cm)

        glPixelTransferf(GL_RED_SCALE,   get_rs)
        glPixelTransferf(GL_GREEN_SCALE, get_gs)
        glPixelTransferf(GL_BLUE_SCALE,  get_bs)
            
        glPixelTransferf(GL_RED_BIAS,   get_rb)
        glPixelTransferf(GL_GREEN_BIAS, get_gb)
        glPixelTransferf(GL_BLUE_BIAS,  get_bb)

        glPixelStorei(GL_PACK_ALIGNMENT, 4) # reset default
        return cc


#########################################
## Helper for PyOpenGL 2  <-> 3
#_HAVE_PY_OPENGL_2 = (OpenGL.__version__[0] == '2')
if OpenGL.__version__[0] == '2':
    def myGL_PixelTransfer(colMap):
        glPixelTransferi(GL_MAP_COLOR, True);
        glPixelMapfv(GL_PIXEL_MAP_R_TO_R, colMap[0] )
        glPixelMapfv(GL_PIXEL_MAP_G_TO_G, colMap[1] )
        glPixelMapfv(GL_PIXEL_MAP_B_TO_B, colMap[2] )
else:
    def myGL_PixelTransfer(colMap):
        mapsize = len(colMap[0])
        glPixelTransferi(GL_MAP_COLOR, True);
        glPixelMapfv(GL_PIXEL_MAP_R_TO_R, mapsize, colMap[0] )
        glPixelMapfv(GL_PIXEL_MAP_G_TO_G, mapsize, colMap[1] )
        glPixelMapfv(GL_PIXEL_MAP_B_TO_B, mapsize, colMap[2] )
        



######################################################################
##
## colMap cm-helper functions - don't need to be in class
##
######################################################################


# 20100510: note: old code 
#            - python actually comes already with an HSV module of some sort
#
#  //www.cs.rit.edu/~ncs/color/t_convert.html 
#  //Color Conversion Algorithms
#
#  //  RGB to HSV & HSV to RGB
#
#  //  The Hue/Saturation/Value model was created by A. R. Smith in 1978. It 
#  //  is based on such intuitive color characteristics as tint, shade and tone 
#  //  (or family, purety and intensity). The coordinate system is cylindrical, 
#  //  and the colors are defined inside a hexcone. The hue value H runs from 
#  //  0 to 360deg. The saturation S is the degree of strength or purity and is 
#  //  from 0 to 1. Purity is how much white is added to the color, so S=1 
#  //  makes the purest color (no white). Brightness V also ranges from 0 to 1, 
#  //  where 0 is the black.
#
#  //  There is no transformation matrix for RGB/HSV conversion, but the algorithm follows:
#
#  // r,g,b values are from 0 to 1
#  // h = [0,360], s = [0,1], v = [0,1]
#  //       if s == 0, then h = -1 (undefined)
#
#  //// When programming in Java, use the RGBtoHSB and HSBtoRGB
#      functions from the java.awt.Color class.

def cm_HSV2RGB(h,s,v):
    if s == 0:
        return (v,v,v) #// achromatic (grey)
    h = h / 60.    # // sector 0 to 5
    i = int( h )
    f = h - i    #        // factorial part of h
    p = v * ( 1. - s )
    q = v * ( 1. - s * f )
    t = v * ( 1. - s * ( 1. - f ) )
    if i == 0:
        return (v,t,p)
    elif i == 1:
        return (q,v,p)
    elif i == 2:
        return (p,v,t)
    elif i == 3:
        return (p,q,v)
    elif i == 4:
        return (t,p,v)
    else: #        // case 5:
        return (v,p,q)

#############################################################################

cms_colnames_255= {
    'white' : (255, 255, 255),
    'red' : (255, 0, 0),
    'yellow' : (255, 255, 128),
    'green' : (0, 255, 0),
    'cyan' : (0, 255, 255),
    'blue' : (0, 0, 255),
    'magenta' : (255, 0, 255),
    'black' : (0, 0, 0),
    'grey' : (128, 128, 128),
    'gray' : (128, 128, 128),
    'orange' : (255, 128, 0),
    'violet' : (128, 0, 255),
    'darkred' : (128, 0, 0),
    'darkgreen' : (0, 128, 0),
    'darkblue' : (0, 0, 128),
    }

# http://simple.wikipedia.org/wiki/List_of_colors 
#      This page was last changed on 31 May 2010, at 04:10.
cms_colnames_simple_wikipedia_org_List_of_colors="""\
Amaranth	#E52B50
Amber	#FFBF00
Aquamarine	#7FFFD4
Azure	#007FFF
Beige	#F5F5DC
Black	#000000
Blue	#0000FF
Blue-green	#0095B6
Blue-violet	#8A2BE2
Brown	#A52A2A
Byzantium	#702963
Carmine	#960018
Cerise	#DE3163
Cerulean	#007BA7
Champagne	#F7E7CE
Chartreuse green	#7FFF00
Coral	#F88379
Crimson	#DC143C
Cyan	#00FFFF
Electric blue	#7DF9FF
Erin	#00FF3F
Gold	#FFD700
Gray	#808080
Green	#00CC00
Harlequin	#3FFF00
Indigo	#4B0082
Ivory	#FFFFF0
Jade	#00A86B
Lavender	#B57EDC
Lilac	#C8A2C8
Lime	#BFFF00
Magenta	#FF00FF
Magenta rose	#FF00AF
Maroon	#800000
Mauve	#E0B0FF
Navy blue	#000080
Olive	#808000
Orange	#FFA500
Orange-red	#FF4500
Peach	#FFE5B4
Persian blue	#1C39BB
Pink	#FFC0CB
Plum	#8E4585
Prussian blue	#003153
Pumpkin	#FF7518
Purple	#800080
Raspberry	#E30B5C
Red	#FF0000
Red-violet	#C71585
Rose	#FF007F
Salmon	#FA8072
Scarlet	#FF2400
Silver	#C0C0C0
Slate gray	#708090
Spring green	#00FF7F
Taupe	#483C32
Teal	#008080
Turquoise	#40E0D0
Violet	#EE82EE
Viridian	#40826D
White	#FFFFFF
Yellow	#FFFF00
"""

cms_colnames = dict(((n,N.array(rgb, float)/255.) for (n,rgb) in cms_colnames_255.iteritems()))

for lll in cms_colnames_simple_wikipedia_org_List_of_colors.splitlines():
    ccc,vvv = lll.split('\t')
    from .useful import strTranslate
    ccc = strTranslate(ccc, delete="- ").lower()
    c = vvv[1:]
    cms_colnames[ccc] = N.array([int(x,16) for x in (c[:2],c[2:4],c[4:])]) / 255. #not defined yet: cm_readColor(vvv)
for rgb in cms_colnames.itervalues():
    rgb.flags.writeable=False
del rgb,lll,ccc,vvv,c,  strTranslate

cms_grey = ['black', 'white']
cms_spectrum = ['darkred', 'red', 'orange', 'yellow', 'green', 'blue',
            'darkblue', 'violet']
cms_blackbody = ['black', 'darkred', 'orange', 'yellow', 'white']
#CHECK - these are unused:
cms_redgreen = ['red', 'darkred', 'black', 'darkgreen', 'green']
cms_greenred = ['green', 'darkgreen', 'black', 'darkred', 'red']
cms_twocolorarray = ['green', 'yellow', 'red']
cms_spectrum2 = ['darkred', 'red', 'orange', '255:255:0', 'green', 'cyan', 'blue',
                'darkblue', 'violet']
cms_spectrum3 = ['darkred', 'red', 'orange', '255:255:0', 'green', 'cyan', 'blue',
                'darkblue', 'violet', 'white'] # , "200:200:200"
cms_spectrum4 = ['black', 'darkred', 'red', 'orange', '255:255:0', 'green', 'cyan', 'blue',
                'darkblue', 'violet', 'white'] # , "200:200:200"


#############################################################################
def cm_log(n=256):
    """
    return "log"-colormap
    
    shape 3,n
    """
    colMap = N.empty(shape = (3,n), dtype = N.float32)
    colMap[:] = 1. - N.log10(N.linspace(1./n,1., num=n, endpoint=True)) / N.log10(1./n)
    return colMap

def cm_grayMinMax(minCol=(0,0,255), maxCol=(255,0,0), n=256):
    """
    return "GrayMinMax"-colormap
    
    shape 3,n

    1.) set col map to gray,
    2.) set first entry to minCol, last entry to maxCol
    """
    colMap = N.empty(shape = (3,n), dtype = N.float32)
    colMap[:] = N.linspace(0,1,num=n,endpoint=True)
    colMap[:,0] = minCol
    colMap[:,-1] = maxCol
    return colMap

import re
_col_regex = re.compile(r'(\d+)[:,](\d+)[:,](\d+)')
del re

def cm_readColor(c):
    """
    returns RGB tuple (ndarray!) each value between 0..1 
    `c` can be a string like "r:g:b"-string like "0:0:122" (r,g,b in 0..255)
            or a string like "r,g,b"-string
            or a len-3-string like "rgb"-string like F8F    (optionally a starting'#')
            or a len-6-string like "rgb"-string like FF80FF (optionally a starting'#')
            or a string of a pre-defined color-name (ref. `cms_colnames`)
            or a RGB tuple  r,b,b in 0..255  (r,g,b in 0..255)
         (strings are case-INsesitive, '-' and spaces is ignored)

    """
    if not isinstance(c, basestring):
        if len(c) != 3:
            raise ValueError, "tuple/list must be of length 3"
        return N.array(c)
    else:
        if c[0] == '#':
            c=c[1:]
        if '-' in c or ' ' in c:
            from .useful import strTranslate
            c = strTranslate(c, delete="- ")
        c = c.lower()
        mat = _col_regex.match(c)
        if mat:
            return N.array( map(int,mat.groups()) ,dtype=N.float ) / 255.
        if c in cms_colnames:
            return cms_colnames[c].copy() # !!!! make copy otherwise the original would get corrupted
        elif len(c)==3:
            return N.array([int(x,16) for x in c]) / 255.
        elif len(c)==6:
            return N.array([int(x,16) for x in (c[:2],c[2:4],c[4:])]) / 255.
        else:
            raise ValueError("%s not understood as color"%(c,))

def cm_calcDiscreteCM(colseq={30: "blue", 80:"red"}, refMaxVal=100, n=256):
    """
    calculates a colormap of length `n` - r,g,b are between 0..1
    starting with black (0,0,0)
    `colseq`: dictionary of color steps. a color step is defined as (value, color) pair
              color can be anything understood by `cm_readColor()` 
                  (colname-string, "r:g:b"-string, "ff80ff", 3-tuple, ...)
    `refMaxVal`: defines the heighest level the values compared against
                 e.g. if `refMaxVal`=100 the values are in "percent"
                      the 100% level refers to to the last (`n-1`) colmap entry,

    transitions are discrete
    returns array of shape 3,n
    """
    colMap = N.zeros(shape=(3, n), dtype=N.float32)
    for stepVal,col in sorted(colseq.iteritems()):
        i0 = int(float(stepVal)/refMaxVal * (n-1))
        colMap[:, i0:] = cm_readColor( col )[:,None]
    return colMap

def cm_calcSmoothCM(colseq=['darkred', 'darkgreen'], reverse=False, n=256):
    """
    calculates a colormap of length `n` - r,g,b are between 0..1
    return colormap created from list-of-colorNames
           instead of a color-name a "r:g:b"-string like "0:0:122"
           you can use predefined name-lists, they start "cms_"
               like cms_redgreen

    transitions are linearly smooth
    returns array of shape 3,n
    """

    if reverse:
        colseq = colseq[:]
        colseq.reverse()

    colMap = N.zeros(shape=(3, n), dtype=N.float32)
    nColNames = len(colseq)
    #  print nColNames
    c = 0
    acc = cm_readColor( colseq[0] )
    # print acc
    for i in range( 0, nColNames-1 ):
        rgb0 = cm_readColor( colseq[i] )
        rgb1 = cm_readColor( colseq[i+1] )
        delta = rgb1 - rgb0

        # print "===> ", i, colseq[i], colseq[i+1], "  ", rgb0, rgb1, "   d: ", delta

        sub_n_f = n / (nColNames-1.0)
        sub_n   = int(n / (nColNames-1))
        # print "*****    ", c, "  ", i*sub_n_f, " ", i*sub_n,  " ++++ ", int( i*sub_n_f+.5 )

        if int( i*sub_n_f+.5 ) > c:
            sub_n += 1             # this correct rounding - to get
            #              correct total number of entries
        delta_step = delta / sub_n
        for i in range(sub_n):
            # print c, acc
            colMap[:, c] = acc

            c+=1
            acc += delta_step
    if(c < n):
        #  print c, acc
        colMap[:, c] = acc
        #      else:
        #          print "** debug ** c == self.cm_size ..."

    return colMap

def cm_grey(reverse=False, n=256):
    """
    return "grey"-colormap (made using cm_calcSmoothCM(cms_grey))
    
    shape 3,n
    """
    return cm_calcSmoothCM(cms_grey, reverse, n)

def cm_col(reverse=False, n=256):
    """
    return "spectrum3"-colormap
    
    shape 3,n
    """

    return cm_calcSmoothCM(cms_spectrum3, reverse, n)

def cm_blackbody(reverse=False, n=256):
    """
    return "blackbody"-colormap
    
    shape 3,n
    """
    return cm_calcSmoothCM(cms_blackbody, reverse, n)

def cm_wheel(cycles=1, blackZero=None, n=256):
    """
    return "cycling-HSV-colors"-colormap
         (red-green-blue-red)
    
    shape 3,n

    if `blackZero` is True:
         set first entry ("background") to zero
    `blackZero` is None: 
         do blackZero if cycles==1
    """


    colMap = N.empty(shape=(3, n), dtype=N.float32)
    for i in range( n ):
        colMap[:,i] = cm_HSV2RGB( (cycles* i * 360./256 ) % 360., 1, 1)
    if blackZero or (blackZero is None and cycles>1):
        colMap[:,0] = 0

    return colMap

def cm_gray(gamma=1, n=256):
    """
    return "gray"-colormap, respective a gamma value
        (cm_calcSmoothCM() is not used, but N.linspace() (for gamma==1))
    shape 3,n
    """
    if gamma == 1:
        colMap = N.empty(shape = (3,n), dtype = N.float32)
        colMap[:] = N.linspace(0,1,num=n,endpoint=True)
    else:
        gamma = float(gamma)
        wmax = 0 + (1 - 0) * ((n - 0) / (1 - 0)) ** gamma
        colMap = N.empty(shape = (3,n), dtype = N.float32)
        colMap[:] = \
              (0 + (1 - 0) * ((N.arange(n) - 0) / (1 - 0)) **gamma) / wmax

    return colMap
