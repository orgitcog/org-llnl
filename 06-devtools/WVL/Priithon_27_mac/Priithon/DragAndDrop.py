"""
Priithon simple drag-and-drop window
"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

import wx

#----------------------------------------------------------------------

class ClipTextPanel(wx.Panel):
    def __init__(self, parent, log):
        wx.Panel.__init__(self, parent, -1)

        #self.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, False))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, -1,
                               "Copy/Paste text to/from\n"
                               "this window and other apps"), 0, wx.EXPAND|wx.ALL, 2)

        self.text = wx.TextCtrl(self, -1, "", style=wx.TE_MULTILINE|wx.HSCROLL)
        sizer.Add(self.text, 1, wx.EXPAND)

        hsz = wx.BoxSizer(wx.HORIZONTAL)
        hsz.Add(wx.Button(self, 6050, " Copy "), 1, wx.EXPAND|wx.ALL, 2)
        hsz.Add(wx.Button(self, 6051, " Paste "), 1, wx.EXPAND|wx.ALL, 2)
        sizer.Add(hsz, 0, wx.EXPAND)
        sizer.Add(wx.Button(self, 6052, " Copy Bitmap "), 0, wx.EXPAND|wx.ALL, 2)

        wx.EVT_BUTTON(self, 6050, self.OnCopy)
        wx.EVT_BUTTON(self, 6051, self.OnPaste)
        wx.EVT_BUTTON(self, 6052, self.OnCopyBitmap)

        self.SetAutoLayout(True)
        self.SetSizer(sizer)


    def OnCopy(self, evt):
        self.do = wx.TextDataObject()
        self.do.SetText(self.text.GetValue())
        if not wx.TheClipboard.Open():
            raise RuntimeError, "cannot open clipboard"
        wx.TheClipboard.SetData(self.do)
        wx.TheClipboard.Close()


    def OnPaste(self, evt):
        do = wx.TextDataObject()
        if not wx.TheClipboard.Open():
            raise RuntimeError, "cannot open clipboard"
        success = wx.TheClipboard.GetData(do)
        wx.TheClipboard.Close()
        if success:
            self.text.SetValue(do.GetText())
        else:
            wx.MessageBox("There is no data in the clipboard in the required format",
                         "Error")

    def OnCopyBitmap(self, evt):
        dlg = wx.FileDialog(self, "Choose a bitmap to copy", wildcard="*.bmp")
        if dlg.ShowModal() == wx.ID_OK:
            bmp = wx.Bitmap(dlg.GetFilename(), wx.BITMAP_TYPE_BMP)
            bmpdo = wx.BitmapDataObject(bmp)
            if not wx.TheClipboard.Open():
                raise RuntimeError, "cannot open clipboard"
            wx.TheClipboard.SetData(bmpdo)
            wx.TheClipboard.Close()

            wx.MessageBox("The bitmap is now in the Clipboard.  Switch to a graphics\n"
                         "editor and try pasting it in...")
        dlg.Destroy()

#----------------------------------------------------------------------

class MyFileDropTarget(wx.FileDropTarget):
    def __init__(self, window, execStr, execModule, redirStdOut, callAfter=False):
        wx.FileDropTarget.__init__(self)

        if execModule is None:
            import __main__
            execDict = __main__.__dict__
        else:
            execDict = execModule
        self.window = window
        self.execStr = execStr
        self.execDict = execDict
        self.redirStdOut = redirStdOut
        self.callAfter   = callAfter

    def OnDropFiles(self, x, y, filenames):
        import sys
        locals = { "fns": filenames,
                   "fn": filenames[0],
                   }
        try:
            locals["dropTextCtrl"] = self.window.text
        except AttributeError:
            pass

        def fff():
            if self.redirStdOut:
                stdout = sys.stdout
                try:
                    sys.stdout = self.window.text
                except AttributeError:
                    pass
            try:
                if callable(self.execStr):
                    self.execStr( filenames )
                else:
                    exec self.execStr in self.execDict, locals
            finally:
                if self.redirStdOut:
                    sys.stdout = stdout

        if self.callAfter:
            wx.CallAfter( fff )
        else:
            fff()

        """
        self.window.text.SetInsertionPointEnd()

        #          self.window.WriteText("\n%d file(s) dropped at %d,%d:\n" %
        #                                (len(filenames), x, y))

        for file in filenames:
            global fn
            #see email
            #Re: [wxPython-users] wxFileDropTarget get filename %-encoded (on gtk not on msw)
            #From: Robin Dunn <robin@alldunn.com>
            #To: wxPython-users@lists.wxwidgets.org
            #Date: Wednesday 12:33:49 pm
            if wx.Platform == "__WXGTK__" and wx.VERSION[:2] == (2,4):
                import urllib
                fn = urllib.unquote(file)
            else:
                fn = file
            self.window.WriteText("Open: "+ fn + '\n')
            from . import usefulX as Y
            from . import Mrc
            global v, a
            originLeftBottom=None
            try:
                try:
                    import useful as U
                    a = U.loadImg(fn)
                    originLeftBottom=0
                except IOError:
                    a = Mrc.bindFile(fn)
            except:
                import sys
                e = sys.exc_info()
                self.window.text.WriteText("Error when opening: %s - %s" %\
                                      (str(e[0]), str(e[1]) ))
            else:
                Y.view(a, originLeftBottom=originLeftBottom)
        """

class FileDropPanel(wx.Panel):
    def __init__(self, parent, execStr, execModule, redirStdOut, callAfter, caption):
        wx.Panel.__init__(self, parent, -1)

        #self.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, False))
        sizer = wx.BoxSizer(wx.VERTICAL)
        if caption:
            sizer.Add(wx.StaticText(self, -1, caption),
                      0, wx.EXPAND|wx.ALL, 2)

        self.text = wx.TextCtrl(self, -1, "",
                                style = wx.TE_MULTILINE|wx.HSCROLL)
        self.dropTarget = MyFileDropTarget(self, execStr, execModule, redirStdOut, callAfter)
        self.text.SetDropTarget(self.dropTarget)
        sizer.Add(self.text, 1, wx.EXPAND)

        self.SetAutoLayout(True)
        self.SetSizer(sizer)


def DropFrame(
    execStr="print fns",
    caption="Drop some file(s) here:",
    title="Priithon file drop", 
    redirStdOut=True,
    callAfter=False,
    execModule=None,
    frame=None,
    panel=None
    ):
    """
    open a new wxFrame containing a text control
    above a caption string is printed (if caption is not '')
    execStr will be executed for every drag&drop operation
    it gets evaluated in `execModule` + some extra var names:
        fns = list of dropped filenames
        fn  = first (or mostly only) file being dropped
        drop0TextCtrl = wxTextControl (inside a panel inside the frame)
    IF `execStr` is a callable it gets called direct with fns as argument 
        `execModule` is then ignored
    if redirStdOut is True, set the text-control to be stdout 
           (and reset back after executing)
    if callAfter:
        use wx.CallAfter to release drag&drop mouse capture 
        before calling execStr (asynchronously!)
    if execModule is None: use __main__
    if frame is None, create a new frame using title as frame title
        otherwise use frame as parent for the dropPanel
    if `panel` is not None: 
        use `panel` as drop target and ignore `frame` and `title`
    """

    if panel is None:
        newFrame=False
        if frame is  None:
            frame = wx.Frame(None, -1, title)
            newFrame=True
        dd = FileDropPanel(frame, execStr, execModule, redirStdOut, callAfter, caption)
        if newFrame:
            frame.Show()
    else:
        if hasattr(panel, "__contains__"):
            for p in panel:
                dropTarget = MyFileDropTarget(p, execStr, execModule, redirStdOut, callAfter)
                p.SetDropTarget(dropTarget)
        else:
            dropTarget = MyFileDropTarget(panel, execStr, execModule, redirStdOut, callAfter)
            panel.SetDropTarget(dropTarget)
        
