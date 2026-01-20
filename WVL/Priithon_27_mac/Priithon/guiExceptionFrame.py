"""
Priithon excetions show gui exception frame
"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

import wx
import traceback

id_close = wx.NewId()
id_print = wx.NewId()
id_debug = wx.NewId()

numberOfOpenExcWindows = 0

class MyFrame(wx.Frame):
    def __init__(self, exctype, value, tb, title="Uncaught exception occured", parent=None):
        wx.Frame.__init__(self,parent, -1, title, size=(600,600), style=wx.DEFAULT_FRAME_STYLE|wx.CENTER)

        self.exctype = exctype
        self.value   = value
        self.tb      = tb
        self.resetStdOut = None
        #ExcType = str(exctype)
        #ExcVal  = str(value)
        EStr = traceback.format_exception_only(exctype, value)[0]
        # >>> traceback.format_exception_only.__doc__
        # 'Format the exception part of a traceback.
        
        #     The arguments are the exception type and value such as given by
        #     sys.last_type and sys.last_value. The return value is a list of
        #     strings, each ending in a newline.
        
        #     Normally, the list contains a single string; however, for
        #     SyntaxError exceptions, it contains several lines that (when
        #     printed) display detailed information about where the syntax
        #     error occurred.
        
        #     The message indicating which exception occurred is always the last
        #     string in the list.

        ExcStr = '\n********************************************\n'.join(traceback.format_exception(exctype, value, tb, limit=None))

        self.sizer = wx.BoxSizer(wx.VERTICAL)

#         hs = wx.BoxSizer(wx.HORIZONTAL)
#         self.sizer.Add(hs, 0, wx.EXPAND)
#         hs.Add(wx.StaticText(self, -1, "Exc Type:"), 0, wx.ALL, 5)
#         self.txtZ = wx.TextCtrl(self, -1, ExcType, size=(40,-1))
#         #wx.EVT_TEXT(self, self.txtZ.GetId(), self.OnTxtZ)
#         hs.Add(self.txtZ, 1)

#         hs = wx.BoxSizer(wx.HORIZONTAL)
#         self.sizer.Add(hs, 0, wx.EXPAND)
#         hs.Add(wx.StaticText(self, -1, "Exc Value:"), 0, wx.ALL, 5)
#         self.txtZ = wx.TextCtrl(self, -1, ExcVal, size=(40,-1))
#         #wx.EVT_TEXT(self, self.txtZ.GetId(), self.OnTxtZ)
#         hs.Add(self.txtZ, 1)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(hs, 0, wx.EXPAND)
        hs.Add(wx.StaticText(self, -1, "Exception:"), 0, wx.ALL, 5)
        style = wx.TE_MULTILINE if EStr.count('\n')>1 else 0
        self.txtZ = wx.TextCtrl(self, -1, EStr, size=(40,-1), style=style)
        #wx.EVT_TEXT(self, self.txtZ.GetId(), self.OnTxtZ)
        hs.Add(self.txtZ, 1)


        #hs = wx.BoxSizer(wx.HORIZONTAL)
        #self.sizer.Add(hs, 1, wx.EXPAND)

        tc = wx.TextCtrl(self, -1, ExcStr, style=wx.TE_MULTILINE)
        tc.SetInsertionPointEnd() # scroll to end of the traceback (Traceback (most recent call last):)
        #hs.Add(tc, 1, wx.ALL, 2)
        self.sizer.Add(tc, 1, wx.EXPAND)

        hs = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(hs, 0, wx.EXPAND)
        b = wx.Button(self, id_close, "dismiss [ESC]")
        wx.EVT_BUTTON(self, id_close, self.OnClose)
        hs.Add(b, 0, wx.ALL, 5)

        b = wx.Button(self, id_print, "print to stderr && dismiss")
        wx.EVT_BUTTON(self, id_print, self.OnPrint)
        hs.Add(b, 0, wx.ALL, 5)

        b = wx.Button(self, id_print, "email bug report")
        wx.EVT_BUTTON(self, id_print, self.OnEmail)
        hs.Add(b, 0, wx.ALL, 5)

        b = wx.Button(self, id_debug, "debug")
        wx.EVT_BUTTON(self, id_debug, self.OnDebug)
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

        global numberOfOpenExcWindows
        numberOfOpenExcWindows +=1
        self.Center()
        self.Show()
        wx.CallAfter(self.Raise) # just to be sure ...
        wx.CallAfter(tc.SetFocus) # just to be sure ... LinuxGTK (so that pressing [ESC] will close)
        
    def OnClose(self, ev):
        del self.exctype, self.value, self.tb
        if self.resetStdOut is not None:
            import sys
            sys.stdout = self.resetStdOut # check if this is smart            
        self.Destroy()
        global numberOfOpenExcWindows
        numberOfOpenExcWindows -=1

    def OnEmail(self, ev):
        import sys, StringIO
        #print >>sys.stderr, "TXT\nTXT"

        #sys.__excepthook__(self.exctype, self.value, self.tb)

        from . import useful as U
        from . import usefulX as Y
        from . import PriDocs

        #subject = Y.wx.GetTextFromUser("Subject line - short error description:", "Subject")
        subject = "[priithonBUG]: " # + subject
        to = "seb.haase+priithonbug@gmail.com"
        # message = ('  ------------------------ system about --------------------------\n'+
        #            PriDocs.helpAbout[PriDocs.helpAbout.index('Priithon version'):]        +
        #            '  ------------------------ system about --------------------------\n')
        # Priithon version: the 3 newest py files are
        # 'guiExceptionFrame.py' dated Wed May  5 15:13:18 2010
        # 'viewerCommon.py' dated Wed May  5 14:55:33 2010
        # 'useful.py' dated Tue Apr 27 14:27:04 2010
        # Priithon package base dir:
        # '/home/shaase/Priithon_25_lin64/Priithon'

        # Platform: linux2
        # Python Version: 2.5.5
        # wxPython Version: 2.8.9.2
        # (wxGTK, unicode, gtk2, wx-assertions-off, SWIG-1.3.29)

        message = """
  --------------------- long explanation -------------------------


  ------------------------ system about --------------------------
%s

  ------------------------   traceback   -------------------------
%s
""" %(
            PriDocs.helpAbout[PriDocs.helpAbout.index('Priithon version'):],
            ''.join(traceback.format_exception(self.exctype, self.value, self.tb))
)

        #password=Y.wx.GetPasswordFromUser("Email server password:", "Password")
        Y.email(message, to, subject=subject, size=(600,400))
        self.Close()

        
    def OnPrint(self, ev):
        import sys
        #print >>sys.stderr, "TXT\nTXT"
        sys.__excepthook__(self.exctype, self.value, self.tb)
        self.Close()

    def OnDebug(self, ev):
        import pdb, sys
        self.resetStdOut = sys.stdout # check if this is smart
        sys.stdout = sys.app.frame.shell
        pdb.pm()
        #self.Close()

    def OnKeyDown(self, event):
        keycode = event.GetKeyCode()
        #print keycode, event.CmdDown(), event.ControlDown()
        if keycode == wx.WXK_ESCAPE:
            self.Close()
        elif keycode == ord('W') and event.CmdDown():
            self.Close()
        else:
            event.Skip()
