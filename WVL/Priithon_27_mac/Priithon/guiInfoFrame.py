"""
Priithon excetions show gui exception frame
"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

import wx
import wx.html

id_close = wx.NewId()
id_print = wx.NewId()
#id_debug = wx.NewId()


class HtmlWindow(wx.html.HtmlWindow):
    """
    from http://wiki.wxpython.org/wxPython%20by%20Example
    """

    def __init__(self, parent, id, size=wx.DefaultSize):
        wx.html.HtmlWindow.__init__(self,parent, id, size=size)
        if "gtk2" in wx.PlatformInfo:
            self.SetStandardFonts()

    def OnLinkClicked(self, link):
        wx.LaunchDefaultBrowser(link.GetHref())


class HtmlInfoFrame(wx.Frame):
    def __init__(self, htmlTxt, title="info", size=(600,600), parent=None):
        wx.Frame.__init__(self,parent, -1, title, size=size, style=wx.DEFAULT_FRAME_STYLE|wx.CENTER)

        self.htmlTxt = htmlTxt
        self.sizer = wx.BoxSizer(wx.VERTICAL)

# #         hs = wx.BoxSizer(wx.HORIZONTAL)
# #         self.sizer.Add(hs, 0, wx.EXPAND)
# #         hs.Add(wx.StaticText(self, -1, "Exc Type:"), 0, wx.ALL, 5)
# #         self.txtZ = wx.TextCtrl(self, -1, ExcType, size=(40,-1))
# #         #wx.EVT_TEXT(self, self.txtZ.GetId(), self.OnTxtZ)
# #         hs.Add(self.txtZ, 1)

# #         hs = wx.BoxSizer(wx.HORIZONTAL)
# #         self.sizer.Add(hs, 0, wx.EXPAND)
# #         hs.Add(wx.StaticText(self, -1, "Exc Value:"), 0, wx.ALL, 5)
# #         self.txtZ = wx.TextCtrl(self, -1, ExcVal, size=(40,-1))
# #         #wx.EVT_TEXT(self, self.txtZ.GetId(), self.OnTxtZ)
# #         hs.Add(self.txtZ, 1)

#         hs = wx.BoxSizer(wx.HORIZONTAL)
#         self.sizer.Add(hs, 0, wx.EXPAND)
#         hs.Add(wx.StaticText(self, -1, "Exception:"), 0, wx.ALL, 5)
#         style = wx.TE_MULTILINE if EStr.count('\n')>1 else 0
#         self.txtZ = wx.TextCtrl(self, -1, EStr, size=(40,-1), style=style)
#         #wx.EVT_TEXT(self, self.txtZ.GetId(), self.OnTxtZ)
#         hs.Add(self.txtZ, 1)


        #hs = wx.BoxSizer(wx.HORIZONTAL)
        #self.sizer.Add(hs, 1, wx.EXPAND)

        #self.tc = wx.TextCtrl(self, -1, infoTxt, style=wx.TE_MULTILINE)
        self.tc = HtmlWindow(self, -1)
        self.tc.SetPage(self.htmlTxt)
        #tc.SetInsertionPointEnd()
        #hs.Add(tc, 1, wx.ALL, 2)
        self.sizer.Add(self.tc, 1, wx.EXPAND)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(hs, 0, wx.EXPAND)
        b = wx.Button(self, id_close, "dismiss [ESC]")
        wx.EVT_BUTTON(self, id_close, self.OnClose)
        hs.Add(b, 0, wx.ALL, 5)

        b = wx.Button(self, id_print, "print to stderr && dismiss")
        wx.EVT_BUTTON(self, id_print, self.OnPrint)
        hs.Add(b, 0, wx.ALL, 5)

        
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        for w in self.GetChildren():
            w.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        wx.EVT_CLOSE(self, self.OnClose)
        #self.sizer.Fit(self)

        self.SetAutoLayout(True)
        self.SetSizer(self.sizer)

        #slider.SetBackgroundColour(wx.LIGHT_GREY)
        #self.SetBackgroundColour(wx.LIGHT_GREY)

        self.Center()
        self.Show()
        wx.CallAfter(self.Raise) # just to be sure ...
        wx.CallAfter(self.tc.SetFocus) # just to be sure ... LinuxGTK (so that pressing [ESC] will close)
        
    def OnClose(self, ev):
        self.Destroy()

    def OnPrint(self, ev):
        import sys
        print >>sys.stderr, self.tc.ToText()
        #sys.__excepthook__(self.exctype, self.value, self.tb)
        self.Close()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        #print keycode, event.CmdDown(), event.ControlDown()
        if keycode == wx.WXK_ESCAPE:
            self.Close()
        elif keycode == ord('W') and event.CmdDown():
            self.Close()
        else:
            event.Skip()

def showHtmlInfo(htmlTxt, title="info", size=(600,600), parent=None):
    """
    open new frame with given txt
    `htmlTxt`: html code of text to be shown
              - note that tags like <h1> MUST be followed by a </h1>
    `title`: title of wx-frame

    `parent`: parent of new frame - default: Priithon's shell frame !!!
    """
    if parent is None:
        import sys
        parent = sys.app.frame # .shell.GetTopLevelParent()
    HtmlInfoFrame(htmlTxt, title, size, parent)

