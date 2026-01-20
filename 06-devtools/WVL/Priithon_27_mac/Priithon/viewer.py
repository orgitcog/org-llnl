"""provides the bitmap OpenGL panel for Priithon's ND 2d-section-viewer
"""
from __future__ import absolute_import
__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

#from .viewerCommon import *
#SyntaxError: 'import *' not allowed with 'from .'
#<hack>
from . import viewerCommon
for n in viewerCommon.__dict__:
    if not n.startswith('_'):
        exec "%s = viewerCommon.%s" % (n,n)
del n, viewerCommon
#</hack>



class GLViewer(GLViewerCommon):
    def __init__(self, parent, imgArr, size=wx.DefaultSize, originLeftBottom=None):
        GLViewerCommon.__init__(self, parent, size)

        self.m_viewComplexAsAbsNotPhase = True
        #20071114 self.m_imgToDo = imgArr
        self.m_imgChanged = True
        #20071114 self.m_histScaleChanged = True

        self.m_imgArr = imgArr
        self.pic_nx = 0  # size as used in current texture
        self.pic_ny = 0
        self.colMap = None
        self.colMap_menuIdx = 0
        self.m_minHistScale = 0
        self.m_maxHistScale = 100
        self.transferf = '' # if !='': exec this string [pixelwise 
        #           (more precise: "vectorized" on all pixels (2d array) "in parallel")]
        #          with x0,x1 being the left/right hist brace as float() !!
        #          and x the pixel value
        #          result must be assigned to varname `y`
        # if result y is a tuple it interpreted as 3 separated "transfer images" for R,G,B 
        # if result y is ndim 3 (shape: 3,ny,nx)[3 for R,G,and B] : y gets transposed to (ny,nx,3) and interpreted as RGB 
        self.transferf_usingXX = False # if True, provide: xx = N.clip((x-x0)/(x1-x0), 0, 1)
        
        self.m_gllist = None
        self.m_texture_list = None

        self.m_gllist_Changed = False # call defGlList() from OnPaint

        if originLeftBottom is None:
            if imgArr.shape[1] == imgArr.shape[0]//2 + 1:
                self.m_originLeftBottom = 8  # output from real_fft2d --- data nx is half-sized
            else:
                try:
                    if not imgArr.meta.originLeftBottom:
                        self.m_originLeftBottom = 0
                    else:
                        self.m_originLeftBottom = 1
                except AttributeError:
                    from . import PriConfig
                    self.m_originLeftBottom = PriConfig.viewerOriginLeftBottomDefault
        else:
            self.m_originLeftBottom = originLeftBottom
            
        if self.m_originLeftBottom==0:
            self.m_aspectRatio *= -1    # !

        if not wx.Platform == '__WXMSW__': #20070525-black_on_black on Windows
            self.SetCursor(wx.CROSS_CURSOR)

        wx.EVT_PAINT(self, self.OnPaint)
        
        #EVT_MIDDLE_DOWN(self, self.OnMiddleDown)
        self.MakePopupMenu()
        ###          wx.Yield() # setImage has gl-calls - so lets make the window first...
        ###          self.setImage(imgArr)



        
        
    def MakePopupMenu(self):
        """Make a menu that can be popped up later"""

        self.m_menu_save = wx.Menu()
        self.m_menu_save.Append(Menu_Save,    "save 2d sec into file")
        self.m_menu_save.Append(Menu_SaveScrShot, "save 2d screen shot 'as seen'")
        self.m_menu_save.Append(Menu_SaveClipboard, "save 2d screen shot 'as seen' into clipboard\tCtrl-C")
        self.m_menu_save.Append(Menu_Assign,  "assign 2d sec to a var name")
        #self.m_menu_save.Append(Menu_SaveND,  "save nd stack into file")

        self.m_menu_colmap = wx.Menu()
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[0],    "gray (linear)")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[1],    "gray (logarithmic)")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[2],    "rainbow")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[3],    "blackbody")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[4],    "rainbow-cycle")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[5],    "rainbow-fastCycle")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[6],    "gray-Min-Max")
        self.m_menu_colmap.AppendRadioItem(Menu_ColMap[7],    "gamma... (GUI)")
        #self.m_menu_save.Append(Menu_SaveND,  "save nd stack into file")
        for i in range(len(Menu_ColMap)):
            wx.EVT_MENU(self, Menu_ColMap[i],      self.OnMenuColMap)

        self.m_menu = wx.Menu()

        self.m_menu.Append(Menu_Zoom2x,    "&zoom 2x\td")
        self.m_menu.Append(Menu_Zoom_5x,   "z&oom .5x\th")
        self.m_menu.Append(Menu_ZoomReset, "zoom &reset\t0")
        self.m_menu.Append(Menu_ZoomCenter,"zoom &center\t9")
        self.m_menu.Append(Menu_Zoom1,"zoom &1\t1")
        self.m_menu.Append(Menu_chgOrig, "c&hange orientation\to")
        self.m_menu.Append(Menu_Reload, "reload\tr")
        #20051116 self.m_menu.Append(Menu_Color, "change ColorMap")
        self.m_menu.AppendMenu(wx.NewId(), "color map\tc", self.m_menu_colmap)

        
        #20050726 self.m_menu.Append(Menu_Save, "save2d")
        self.m_menu.AppendMenu(wx.NewId(), "save", self.m_menu_save)
        #self.m_menu.Append(Menu_Save3d, "save3d")
        self.m_menu.Append(Menu_aspectRatio, "change aspect ratio")
        self.m_menu.Append(Menu_rotate, "display rotated...")
        self.m_menu.Append(Menu_noGfx, "hide all gfx\tb", '',wx.ITEM_CHECK)

        wx.EVT_MENU(self, Menu_ZoomCenter, self.OnCenter)
        wx.EVT_MENU(self, Menu_ZoomOut, self.OnZoomOut)
        wx.EVT_MENU(self, Menu_ZoomIn, self.OnZoomIn)
        wx.EVT_MENU(self, Menu_Zoom2x,     self.OnMenu)
        #wx.EVT_MENU(self, Menu_ZoomCenter, self.OnMenu)
        wx.EVT_MENU(self, Menu_Zoom_5x,    self.OnMenu)
        wx.EVT_MENU(self, Menu_ZoomReset,  self.doReset) # OnMenu)
        wx.EVT_MENU(self, Menu_Zoom1,  lambda ev: self.zoom(1)) # OnMenu)
        wx.EVT_MENU(self, Menu_Color,      self.OnColor)
        wx.EVT_MENU(self, Menu_Reload,      self.OnReload)
        wx.EVT_MENU(self, Menu_chgOrig,      self.OnChgOrig)
        wx.EVT_MENU(self, Menu_Save,      self.OnSave)
        wx.EVT_MENU(self, Menu_SaveScrShot,      self.OnSaveScreenShort)
        wx.EVT_MENU(self, Menu_SaveClipboard,    self.OnSaveClipboard)
        wx.EVT_MENU(self, Menu_Assign,      self.OnAssign)
        wx.EVT_MENU(self, Menu_aspectRatio,      self.OnAspectRatio)
        wx.EVT_MENU(self, Menu_rotate,      self.OnRotate)
        wx.EVT_MENU(self, Menu_noGfx,      self.OnNoGfx)

    def InitGL(self):
        #        // Enable back face culling of polygons based upon their window coordinates.
        #        //glEnable(GL_CULL_FACE)
        #        // Enable two-dimensional texturing.
    
        #glClearColor(1.0, 1.0, 1.0, 0.0)
        from . import PriConfig
        glClearColor(*PriConfig.viewerBkgColor)
#20050520       glEnable(GL_TEXTURE_2D)
        
        #print "InitGL 2"
        """ now in if self.m_doViewportChange:
        glMatrixMode (GL_PROJECTION)
        glLoadIdentity ()
        #//glOrtho (-.375, width-.375, height-.375, -.375, 1., -1.)
        glOrtho (-.375, self.m_w-.375, -.375, self.m_h-.375, 1., -1.)
        #    //  The subtraction of .375 accounts for differences in rasterization of points,
        #    //  lines, and filled primitives in order to ensure their vertices map to pixel
        #    //  coordinates. The reversal of the Y coordinates should buy you a mapping to
        #    //  your inverted window coordinate space in which (0,0) is at the upper left
        #    //  corner.
        #  seb : WE DON'T WANT INVERSION THOUGH !!!!
        glMatrixMode (GL_MODELVIEW)
        glLoadIdentity ()
        #  //glViewport (0, 0, winWidth, winHeight)
        glViewport (0, 0, self.m_w, self.m_h)
        """
        #print "InitGL 3",     self.m_w, self.m_h    
        
        
        #print "InitGL 5"
        self.m_glInited = True

    def InitTex(self):
        from . import PriConfig

        #print "1 InitTex"
        self.tex_nx = 2
        while self.tex_nx<self.pic_nx:
            self.tex_nx*=2 # // texture must be of size 2^n
        self.tex_ny = 2
        while self.tex_ny<self.pic_ny:
            self.tex_ny*=2
        self.picTexRatio_x = float(self.pic_nx) / self.tex_nx
        self.picTexRatio_y = float(self.pic_ny) / self.tex_ny

        self.set_current() # 2012 self.SetCurrent(self.context) # 20111103) # 20041014

        if not self.m_gllist:
            self.m_gllist = glGenLists( 2 )
            #print "make new glLists:", self.m_gllist
        if self.m_texture_list:
            glDeleteTextures(self.m_texture_list)#glDeleteTextures silently  ignores  zeros

        self.m_texture_list = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.m_texture_list)
        
        if PriConfig.viewerInterpolationMinifyDefault:
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
        if PriConfig.viewerInterpolationMagnifyDefault:
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
        else:
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
        #    // GL_CLAMP causes texture coordinates to be clamped to the range [0,1] and is
        #    // useful for preventing wrapping artifacts when mapping a single image onto
        #    // an object.
        #    //  //    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP)
        #    //  //    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP)
        
        if self.m_imgArr.dtype.type in (N.uint8,N.bool_):
            glTexImage2D(GL_TEXTURE_2D,0,  GL_RGB, self.tex_nx,self.tex_ny, 0, 
                         GL_LUMINANCE,GL_UNSIGNED_BYTE, None)
        elif self.m_imgArr.dtype.type == N.int16:
            glTexImage2D(GL_TEXTURE_2D,0,  GL_RGB, self.tex_nx,self.tex_ny, 0, 
                         GL_LUMINANCE,GL_SHORT, None)
        elif self.m_imgArr.dtype.type == N.float32:
            glTexImage2D(GL_TEXTURE_2D,0,  GL_RGB, self.tex_nx,self.tex_ny, 0, 
                         GL_LUMINANCE,GL_FLOAT, None)
        elif self.m_imgArr.dtype.type == N.uint16:
            glTexImage2D(GL_TEXTURE_2D,0,  GL_RGB, self.tex_nx,self.tex_ny, 0, 
                         GL_LUMINANCE,GL_UNSIGNED_SHORT, None)

        elif self.m_imgArr.dtype.type in (N.float64,
                                          N.int32, N.uint32, N.int64, N.uint64, N.int0,
                                          N.complex64, N.complex128):
            glTexImage2D(GL_TEXTURE_2D,0,  GL_RGB, self.tex_nx,self.tex_ny, 0, 
                         GL_LUMINANCE,GL_FLOAT, None)

        else:
            self.error = "unsupported data mode"
            raise ValueError, self.error


        cornerOffsetX = -.5
        cornerOffsetY = -.5
        if self.m_originLeftBottom == 7:
            cornerOffsetX -= self.pic_nx // 2
            cornerOffsetY -= self.pic_ny // 2
        if self.m_originLeftBottom == 8:
            cornerOffsetX -= self.pic_nx-1
            cornerOffsetY -= (self.pic_ny-1) // 2

        glNewList( self.m_gllist, GL_COMPILE )
        glTranslate(cornerOffsetX,cornerOffsetY ,0)  # 20080701:  in new coord system, integer pixel coord go through the center of pixel
        glEnable(GL_TEXTURE_2D)#20050520
        glColor3f(1.0, 1.0, 1.0) #//2004/04/09
        ##################################glColor3f(1.0, 0.0, 0.0);
        #  // Use a named texture.
        glBegin(GL_QUADS)
        
        ##BUG - HANGS glBindTexture(GL_TEXTURE_2D, self.m_texture_list)
        #          #  //    glNormal3f( 0.0F, 0.0F, 1.0F)
        
        #          #  //seb TODO  correct for pixel center vs pixel edge -> add .5 "somewhere"
        
        if self.m_originLeftBottom == 7 or \
           self.m_originLeftBottom == 8:
            sx = self.pic_nx
            ox = sx // 2
            sy = self.pic_ny
            oy = sy // 2
            hx = self.picTexRatio_x / 2. # fft half nyquist
            hy = self.picTexRatio_y / 2. # fft half nyquist
            fx = self.picTexRatio_x      # fft full nyquist
            fy = self.picTexRatio_y      # fft full nyquist
            

            #   quadrands(q): 2 | 1
            #                 --+--
            #                 3 | 4
            #


            d = 1 # display offset (how far left-down from center)
            ex = self.picTexRatio_x / sx
            ey = self.picTexRatio_y / sy


            # output from real_fft2d --- data nx is half-sized
            if self.m_originLeftBottom == 8:
                sx = (self.pic_nx-1) * 2
                ox = sx // 2
                hx = self.picTexRatio_x

                # data: dc left bottom(q 1)     display: dc left-down from center !
                # q 3                           q 1
                
                glTexCoord2f( 0, 0);            glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( hx, 0);           glVertex2i  ( sx, oy-d)
                glTexCoord2f( hx, hy+ey);          glVertex2i  ( sx, sy)
                glTexCoord2f( 0, hy+ey);           glVertex2i  ( ox-d, sy)
                
                # q 2                           q 4
                glTexCoord2f( 0, fy);           glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( 0, hy+ey);           glVertex2i  ( ox-d, 0)
                glTexCoord2f( hx, hy+ey) ;         glVertex2i  ( sx, 0)
                glTexCoord2f( hx, fy);          glVertex2i  ( sx, oy-d)
                
                
                # q 2 (real_fft2d) (4)          q 2
                glTexCoord2f( hx-ex, fy);       glVertex2i  ( 0,    oy)
                glTexCoord2f( hx-ex, hy);          glVertex2i  ( 0,    sy)
                glTexCoord2f( ex,    hy);          glVertex2i  ( ox-d, sy)
                glTexCoord2f( ex,    fy);       glVertex2i  ( ox-d, oy)

                # (real_fft2d) just 1 pixel row: y-dc x-neg
                glTexCoord2f( hx-ex, 0);       glVertex2i  ( 0,    oy)
                glTexCoord2f( hx-ex, ey);          glVertex2i  ( 0,    oy-d)
                glTexCoord2f( ex,    ey);          glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( ex,    0);       glVertex2i  ( ox-d, oy)
                
                
                # q 3 (real_fft2d) (1)          q 3
                glTexCoord2f( ex,  ey );          glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( hx-ex,    ey );          glVertex2i  ( 0 , oy-d)
                glTexCoord2f( hx-ex,    hy) ;         glVertex2i  ( 0 , 0)
                glTexCoord2f( ex, hy);          glVertex2i  ( ox-d, 0)
                
            else: # self.m_originLeftBottom == 7:

                # data: dc left bottom(q 1)     display: dc left-down from center !
                # q 3                           q 1
                glTexCoord2f( 0, 0);            glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( hx+ex, 0);           glVertex2i  ( sx, oy-d)
                glTexCoord2f( hx+ex, hy+ey);          glVertex2i  ( sx, sy)
                glTexCoord2f( 0, hy+ey);           glVertex2i  ( ox-d, sy)
    
                # q 2                           q 4
                glTexCoord2f( 0, fy);           glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( 0, hy+ey);           glVertex2i  ( ox-d, 0)
                glTexCoord2f( hx+ex, hy+ey) ;         glVertex2i  ( sx, 0)
                glTexCoord2f( hx+ex, fy);          glVertex2i  ( sx, oy-d)
    
                # q 4                           q 2
                glTexCoord2f( fx, 0);           glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( fx, hy+ey);          glVertex2i  ( ox-d, sy)
                glTexCoord2f( hx+ex, hy+ey);          glVertex2i  ( 0 , sy)
                glTexCoord2f( hx+ex, 0);           glVertex2i  ( 0 , oy-d)
    
                # q 1                           q 3
                glTexCoord2f( fx, fy);          glVertex2i  ( ox-d, oy-d)
                glTexCoord2f( hx+ex, fy);          glVertex2i  ( 0 , oy-d)
                glTexCoord2f( hx+ex, hy+ey) ;         glVertex2i  ( 0 , 0)
                glTexCoord2f( fx, hy+ey);          glVertex2i  ( ox-d, 0)

        else: #20100325 # self.m_originLeftBottom == 0 or 1:
            ###//(0,0) at left bottom  #20100325  until scale might flip it...
            
            glTexCoord2f( 0, 0)
            glVertex2i  ( 0, 0)
            
            glTexCoord2f( self.picTexRatio_x, 0)
            glVertex2i  ( self.pic_nx, 0)
            
            glTexCoord2f( self.picTexRatio_x, self.picTexRatio_y)
            glVertex2i  ( self.pic_nx, self.pic_ny)
            
            glTexCoord2f( 0, self.picTexRatio_y)
            glVertex2i  ( 0, self.pic_ny)

        # else:                          # self.m_originLeftBottom == 0:
        #     ###//(0,0) at left top
        #     glTexCoord2f( 0,             self.picTexRatio_y)
        #     glVertex2i  ( 0,             0)
            
        #     glTexCoord2f( self.picTexRatio_x, self.picTexRatio_y)
        #     glVertex2i  ( self.pic_nx,        0)
            
        #     glTexCoord2f( self.picTexRatio_x, 0)
        #     glVertex2i  ( self.pic_nx,        self.pic_ny)
            
        #     glTexCoord2f( 0,             0)
        #     glVertex2i  ( 0,             self.pic_ny)
            
        #print "InitGL 3"
        glEnd()
        
        glDisable(GL_TEXTURE_2D) #20050520

        glTranslate(-cornerOffsetX,-cornerOffsetY ,0)  # 20080701:  in new coord system, integer pixel coord go through the center of pixel
        glEndList()
        
    def defGlList(self):
        pass
    def updateGlList(self, glCallsFunctions, refreshNow=True):
        if glCallsFunctions is None:
            def x():
                pass
            self.defGlList = x
        else:
            self.defGlList = glCallsFunctions
        self.m_gllist_Changed = True
        if refreshNow:        
            self.Refresh(False)
        
    def addGlList(self, glCallsFunctions):
        if glCallsFunctions is None:
            def x():
                pass
            glCallsFunctions = x
        try:
            self.defGlList.append( glCallsFunctions )
        except:
            self.defGlList = [ self.defGlList, glCallsFunctions ]

        self.m_gllist_Changed = True
        self.Refresh(False)



    def changeHistogramScaling(self, smin=None, smax=None, RefreshNow=True):
        #20101001 if smin != smax:
        if smin is not None:
            self.m_minHistScale = smin
        if smax is not None:
            self.m_maxHistScale = smax

        #20071114 self.m_histScaleChanged = True
        
        self.m_imgChanged=True  #  // as of 20071114 accepted: before: - i would prefer NOT to reload image into GfxCard
        if RefreshNow:
            self.Refresh(False)

    def OnPaint(self, event):
        try:
            dc = wx.PaintDC(self)
        except:
            # this windows is dead !?
            return
        #self.m_w, self.m_h = self.GetClientSizeTuple()
        try:
            if self.m_w <=0 or self.m_h <=0:
                # THIS IS AFTER wx.PaintDC -- OTHERWISE 100% CPU usage
                return
        except AttributeError:
            return
        if self.error:
            return
        #//seb check PrepareDC(dc)
        #20111103 if not self.GetContext():
        #20111103     print "OnPaint GetContext() error"
        #20111103     return
        
        self.set_current() # 2012 self.SetCurrent(self.context) # 20111103 old: self.SetCurrent()
  
        if not self.m_glInited:
            self.InitGL()

        if self.m_doViewportChange:
            glViewport(0, 0, self.m_w, self.m_h)
            glMatrixMode (GL_PROJECTION)
            glLoadIdentity ()
            glOrtho (-.375, self.m_w-.375, -.375, self.m_h-.375, 1., -1.)
            #20080828: put the .375 stuff back --- otherwise some images were shown with "bottom line shown somewhat at top of image" glOrtho (0, self.m_w, 0, self.m_h, 1., -1.)
            glMatrixMode (GL_MODELVIEW)
            self.m_doViewportChange = False

        if self.m_gllist_Changed:
            try:
                self.defGlList
                
                glNewList( self.m_gllist+1, GL_COMPILE )
                try:
                    for gll in self.defGlList:
                        gll()
                except TypeError: #'list' object is not callable
                    self.defGlList()                    
                glEndList()
            except:
                import traceback as tb
                tb.print_exc(limit=None, file=None)
                self.error = "error with self.defGlList()"
                print "ERROR:", self.error
            self.m_gllist_Changed = False

        if self.m_imgChanged:
            if self.transferf:
                try:
                    import __main__
                    x0 = float(self.m_minHistScale)
                    x1 = float(self.m_maxHistScale)
                    if self.transferf_usingXX:
                        loc = {'x':self.m_imgArr, 
                               'x0': x0,
                               'x1': x1,
                               '_':__main__,
                               'xx': N.clip((self.m_imgArr-x0)/(x1-x0), 0, 1)
                               }
                    else:
                        loc = {'x':self.m_imgArr, 
                               'x0': x0,
                               'x1': x1,
                               '_':__main__
                               }

                    exec(self.transferf, 
                                __main__.__dict__, 
                                loc)
                    data = loc['y']
                    if type(data) is tuple:
                        data = N.array(data)
                    if data.ndim == 3:
                        data = data.transpose((1,2,0))
                    #20071114 itSize = data.itemsize
                    #20071114 glPixelStorei(GL_UNPACK_SWAP_BYTES, data.dtype.byteorder != '=')
                    #20071114 glPixelStorei(GL_UNPACK_ALIGNMENT, itSize)
                    fBias = 0
                    f = 1
                except:
                    import sys
                    print >>sys.stderr, "  ### ### cought exception in transfer function:  #### " 
                    import traceback
                    traceback.print_exc()
                    print >>sys.stderr, "  ### ### cought exception in transfer function:  #### " 
                    data = self.m_imgArr
                    fBias = 0
                    f = 1
            else:
                data = self.m_imgArr
            # TODO check inf, nan
            

            
            if data.dtype.type in (N.complex64, N.complex128):
                # handle (non gfx-card) complex dtypes

                if self.m_viewComplexAsAbsNotPhase:
                    data = abs(N.asarray(data, N.float32)) # check if this does temp copy
                else:
                    #from Priithon.all import U
                    #data = U.phase(data.astype(N.float32)
                    #not temp copy for type conversion:
                    data =  N.arctan2(N.asarray(data.imag, N.float32),
                                      N.asarray(data.real, N.float32))

            if bugXiGraphics:
                # if the gfx-card can on handle uint8 dtype

                self.bugXiGraphicsmi = self.m_minHistScale # self.m_imgArr.min()
                self.bugXiGraphicsma = self.m_maxHistScale # self.m_imgArr.max()
                den = (self.bugXiGraphicsma - self.bugXiGraphicsmi)
                if den == 0:
                    den = 1
                self.bugXiGraphicsfa = 255./den

                data = ((data-self.bugXiGraphicsmi) *self.bugXiGraphicsfa)
                data = N.clip(data, 0, 255)
                data = data.astype(N.uint8)

                fBias = 0
                f = 1
            elif not self.transferf:
                # handle non gfx-card dtypes
                if data.dtype.type in (N.float64,
                                       N.int32, N.uint32,N.int64, N.uint64, N.int0):
                    data = data.astype(N.float32)
                
                srange =  float(self.m_maxHistScale - self.m_minHistScale)
                if srange == 0:
                    fBias = 0
                    f = 1
                else:
                    # maxValueWhite : value that represents "maximum color" - i.e. white
                    if data.dtype.type == N.uint16:
                        maxValueWhite  = (1<<16) -1.
                    elif data.dtype.type == N.int16:
                        maxValueWhite  = (1<<15) -1.
                    elif data.dtype.type in (N.uint8,N.bool_):
                        maxValueWhite  = (1<<8) -1.
                    else:
                        maxValueWhite  = 1.

                    fBias =  -self.m_minHistScale / srange
                    f     =  maxValueWhite / srange

            imgType = data.dtype.type
            dataString = data.tostring()            
            itSize = data.itemsize
            
            glPixelStorei(GL_UNPACK_SWAP_BYTES, not data.dtype.isnative)
            glPixelStorei(GL_UNPACK_ALIGNMENT, itSize)
            

            glPixelTransferf(GL_RED_SCALE,   f)
            glPixelTransferf(GL_GREEN_SCALE, f)
            glPixelTransferf(GL_BLUE_SCALE,  f)
            
            glPixelTransferf(GL_RED_BIAS,   fBias)
            glPixelTransferf(GL_GREEN_BIAS, fBias)
            glPixelTransferf(GL_BLUE_BIAS,  fBias)
    
            #self.fBias = fBias  #for debugging
            #self.f     = f      #for debugging
            #self.data = data    #for debugging
            #self.imgType = imgType #for debugging
            #self.itSize  = itSize #for debugging
            
            if self.colMap is not None:
                myGL_PixelTransfer(self.colMap)
            else:
                glPixelTransferi(GL_MAP_COLOR, False);
                # //printf("%8d %9f \n", -min, f)

            if (self.pic_ny, self.pic_nx) != data.shape[:2]: # could be RGB at this point - see transferf
                (self.pic_ny, self.pic_nx) = data.shape[:2]
                ################### self.SetCurrent();            
                self.InitTex()
                # self.SetDimensions(-1,-1, self.pic_nx,self.pic_ny)


            glBindTexture(GL_TEXTURE_2D, self.m_texture_list)

            if data.ndim == 3:
                format = GL_RGB
            else:
                format = GL_LUMINANCE
                

            if imgType in (N.uint8,N.bool_):
                glTexSubImage2D(GL_TEXTURE_2D,0,  0,0,  self.pic_nx,self.pic_ny, 
                                format,GL_UNSIGNED_BYTE, dataString)
            elif imgType == N.int16:
                glTexSubImage2D(GL_TEXTURE_2D,0,  0,0,  self.pic_nx,self.pic_ny, 
                                format,GL_SHORT,         dataString)
            elif imgType == N.float32:
                glTexSubImage2D(GL_TEXTURE_2D,0,  0,0,  self.pic_nx,self.pic_ny, 
                                format,GL_FLOAT,         dataString)
            elif imgType == N.uint16:
                glTexSubImage2D(GL_TEXTURE_2D,0,  0,0,  self.pic_nx,self.pic_ny, 
                                format,GL_UNSIGNED_SHORT, dataString)
            else:
                self.error = "unsupported data mode"
                raise ValueError, self.error
            
            self.m_imgChanged = False

        if self.m_zoomChanged:
            if self.m_x0 is None:
                self.center(refreshNow=False)

            sx,sy = self.m_scale,self.m_scale* self.m_aspectRatio
            glMatrixMode (GL_MODELVIEW)
            glLoadIdentity ()
            glTranslate(self.m_x0, self.m_y0,0)
            
            if self.m_rot: #20100325 
                offX,offY = sx*self.pic_nx/2., sy*self.pic_ny/2.
                glTranslate(offX,offY,0)
                glRotate(self.m_rot, 0,0,1)
                glTranslate(-offX,-offY,0)
            glScaled(sx,sy,1.)
            
            self.m_zoomChanged = False

        #              //CHECK
        #              // clear color and depth buffers
        #              //seb seb sbe seb seb 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glCallList( self.m_gllist )
        glCallList( self.m_gllist+1 )

        if self.m_moreMaster_enabled:
            #  for l,on in zip(self.m_moreGlLists,self.m_moreGlLists_enabled):
            #      if on:
            #          glCallList( l )
            enabledGLlists = [i for (i,on) in zip(self.m_moreGlLists,
                                              self.m_moreGlLists_enabled) if on]
            if len(enabledGLlists):
                glCallLists( enabledGLlists )
        
        glFlush()
        self.SwapBuffers()

    def setImage(self, imgArr):
        self.m_imgArr = imgArr
        self.m_imgChanged=True

        self.Refresh(0)



    def goFFTmode(self):
        self.setOriginLeftBottom(7)

    #20080707 def doOnFrameChange(self):
    #20080707     pass

    #20080707 def doOnMouse(self, x,y, xyEffVal):
    #20080707     # print "xy: --> %7.1f %7.1f" % (x,y)
    #20080707     pass
    
      
    def OnReload(self, event=None):
        self.m_imgChanged = True
        self.Refresh(False)

        
    def OnColor(self, event=None):
        #Windows: works all fine when accel-key used WITH ALT # print "DEBUG: OnColor", self
        if self.colMap_menuIdx == 0:
            self.setColMap(cm_log())
        elif self.colMap_menuIdx == 1:
            self.setColMap( cm_col() )
        elif self.colMap_menuIdx == 2:
            self.setColMap( cm_blackbody() )
        elif self.colMap_menuIdx == 3:
            self.setColMap( cm_wheel() )
        elif self.colMap_menuIdx == 4:
            self.setColMap( cm_wheel(10) )
        elif self.colMap_menuIdx == 5:
            self.setColMap( cm_grayMinMax() )
        elif self.colMap_menuIdx == 6:
            self.setColMap(None)
#         if self.colMap != None:
#             self.cmnone()
#         else:
#             self.cmcol()

        self.colMap_menuIdx += 1
        self.colMap_menuIdx %= 7
        # 20040713 now in cmWheel,cmLog,...   self.updateHistColMap()

        
        
        
    def updateHistColMap(self):
        self.my_hist.colMap = self.colMap
        self.my_hist.m_histScaleChanged = True
        self.my_hist.m_imgChanged = True
        self.my_hist.Refresh(0)
        
        

    def OnMenuColMap(self, ev):
        menuIdx = Menu_ColMap.index(ev.GetId())
        if menuIdx == 7:
            #self.gammawin = GammaPopup(self)
            from . import usefulX as Y
            Y.vGammaGUI(self)
            self.colMap_menuIdx = 7-1
            return
        elif menuIdx == 0:
            self.setColMap(None)
        elif menuIdx == 1:
            self.setColMap( cm_log() )
        elif menuIdx == 2:
            self.setColMap( cm_col() )
        elif menuIdx == 3:
            self.setColMap( cm_blackbody() )
        elif menuIdx == 4:
            self.setColMap( cm_wheel() )
        elif menuIdx == 5:
            self.setColMap( cm_wheel(10) )
        elif menuIdx == 6:
            self.setColMap( cm_grayMinMax() )

        self.colMap_menuIdx = (menuIdx+1)%7
        

    def setColMap(self, colmap):
        """
        `colMap` must be an ndarray of shape (3,n) [n should be 256 - 512 might work on some gfx-cards]
                      and of type float32
            OR `colMap` can be `None` - then OpenGL uses linear (grey) colomap

        new color map is then being used - all needed update-calls are made:
            self.changeHistogramScaling()
            self.updateHistColMap()
        """
        self.colMap = colmap
        self.changeHistogramScaling()
        self.updateHistColMap()    

    ###############3###############3###############3###############3###############3
    ###############3###############3###############3###############3###############3

      

        
    
def view(array, title=None, size=None, parent=None):
    if len(array.shape) != 2:
        raise ValueError, "array must be of dimension 2"

    ### size = (400,400)
    if size is None:
        w,h = (array.shape[1],array.shape[0], )
        if h/2 == (w-1):  ## real_fft2d
            w = (w-1)*2
        if w>600 or h>600:
            size=400
    elif type(size) == int:
        w,h = size, size
    else:
        w,h = size

    if title is None or title =='':
        try:
            fn = self.dataOrig.meta.filename
        except AttributeError:
            title='' 
            pass
        else:
            import os.path
            fn = os.path.basename(fn)
            title = "<%s>" % (fn,)
        
    frame = wx.Frame(parent, -1, title)
    canvas = GLViewer(frame, array, size=(w,h))
    sizer = wx.BoxSizer(wx.VERTICAL)
    sizer.Add(canvas, 1, wx.EXPAND | wx.ALL, 5)
    frame.SetSizer(sizer)
    # sizer.SetSizeHints(frame) 
    #2.4auto  frame.SetAutoLayout(1)
    sizer.Fit(frame)

    frame.Show(1)
    frame.Layout() # hack for Linux-GTK
    return canvas
