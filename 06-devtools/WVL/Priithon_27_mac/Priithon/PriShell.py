"""PyShell is a python shell application."""
from __future__ import absolute_import
#seb: PriShell

# The next two lines, and the other code below that makes use of
# ``__main__`` and ``original``, serve the purpose of cleaning up the
# main namespace to look as much as possible like the regular Python
# shell environment.
#20091208-PyFlakes import __main__
#20090806 original = __main__.__dict__.keys()

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"
#seb __author__ = "Patrick K. O'Brien <pobrien@orbtech.com>"
#seb __cvsid__ = "$Id: PyShell.py,v 1.7 2004/03/15 13:42:37 PKO Exp $"
#seb __revision__ = "$Revision: 1.7 $"[11:-2]

#20091208-PyFlakes import wx


"""
The main() function needs to handle being imported, such as with the
pyshell script that wxPython installs:

    #!/usr/bin/env python

    from wx.py.PyShell import main
    main()
"""

def main(startMainLoop=True, cleanup_main=True):
    """The main function for the PyShell program.

    `cleanup_main`: if True, all but few names are del'ed from `__main__`
                    CHECK / FIXME: `shell` is left in
    """
    import wx

    #20071212: local class, just so that we don't have a "App" in __main__
    class App(wx.App):
        """PyShell standalone application."""

        def OnInit(self):
            import wx,sys,os
            # 20120228 seb
            #import wx.py                        # 20120228 seb
            #import Priithon.py                  # 20120228 seb
            #import sys                          # 20120228 seb
            #sys.modules['Priithon.py'] = wx.py  # 20120228 seb

            #from .py import shell
            # absolute import
            # ValueError: Attempted relative import in non-package
            # NOTE: this is where Priithon is beeing started - i.e. we are not yet "inside" the Priithon module
            from Priithon.py import shell

            #20120229 wx2.9 wx.InitAllImageHandlers()  
            #wx.InitAllImageHandlers is now an empty function that does nothing but exist for backwards compatibility. 
            #The C++ version is now called automatically when wxPython is initialized. Since all the handlers are 
            #included in the wxWidgets shared library anyway, this imposes only a very small amount of overhead 
            #and removes several unneccessary problems.
            title = "priithon on %s (pid: %s)" % (
                wx.GetHostName(), os.getpid())
            print title # to know which terminal window belongs to which Priithon
            self.frame = shell.ShellFrame(
                title=title,
                #20120228 
                introText=' !!! Welcome to Priithon !!! \n'+
                #20120228 
                '(Python %s on %s)' % (sys.version.replace('\n',' '), sys.platform),
                #)#20120228 
                introStatus='Priithon: %s' % sys.argv)
            self.frame.SetSize((750, 525))
            self.frame.Show()
            self.SetTopWindow(self.frame)
            self.frame.shell.SetFocus()
            return True

    import __main__
    md = __main__.__dict__
    #seb keepers = original
    #seb keepers.append('App')
    #print keepers
    #['__builtins__', '__name__', '__file__', '__doc__', '__main__', 'App']
    #seb note: wee don't need to keep any of these 

    # Cleanup the main namespace, (OLD: ...,leaving the App class.)
    if cleanup_main:
        for key in md.keys():
            if key not in [
                #20071212 'App',
                '__author__',
                '__builtins__',
                #'__doc__',
                #'__file__',
                '__license__',
                #'__main__',
                '__name__',
                #'main',
                #'original',
                #20110221 'shell',  # this is used in py/shell.py::shellMessage
                #'wx'
                ]:
                #['App', '__author__', '__builtins__', '__doc__', '__file__', '__license__', '__main__', '__name__', 'main', 'original', 'shell', 'wx']
                del md[key]


    # Mimic the contents of the standard Python shell's sys.path.
    #   python prepends '' if called as 'python'
    #   but    prepends '<path>' if called as 'python <path>/PriShell.py'
    #          in this case we replace sys.path[0] with ''
    import sys
    if sys.path[0]:
        sys.path[0] = ''

    # Create an application instance. (after adjusting sys.path!)
    sys.app = None # dummy to force Priithon.Y getting loaded
    #20090811 app = App(0)
    #       default: wx.App(redirect=False, filename=None, useBestVisual=False,
    #                       clearSigInt=True)
    app = App(redirect=False, filename=None, useBestVisual=False, 
              clearSigInt=False)

    #20070914 del md['App']
    # Add the application object to the sys module's namespace.
    # This allows a shell user to do:
    # >>> import sys
    # >>> sys.app.whatever
    sys.app = app

    #seb: load Priithon modules
    #exec "from Priithon.all import *" in __main__.__dict__
    #U._fixDisplayHook()
    try:
        from Priithon import startupPriithon
    except:
        import traceback
        wx.MessageBox(traceback.format_exc(), 
                      "Exception while Priithon Startup", 
                      wx.ICON_ERROR|wx.OK)

    if len(sys.argv)>1:
        if sys.argv[1] == '-xc':
            sys.argv[1] = '-c'
        if sys.argv[1] == '-c':
            if len(sys.argv)>2:
                def execArgvs(lines):
                    #print shell
                    #print shell.stdout
                    #import sys
                    #print sys.stdout
                    for l in lines:
                        __main__.Y.shellExec( l, prompt=False, verbose=False )
                        #__main__.shell.push( l, newPrompt=False )

                wx.CallAfter(execArgvs, sys.argv[2].splitlines())
                del sys.argv[2]

    # Start the wxPython event loop.
    #del wx
    if startMainLoop:
        app.MainLoop()
        print " * priithon ended. *" # to know when when the process has quit

if __name__ == '__main__':
    import sys
    cleanup_main=True
    try:
        iii = sys.argv.index('--no-cleanup-main')
    except ValueError:
        pass
    else:
        del sys.argv[iii]
        cleanup_main=False
        del iii
    del sys
    main(cleanup_main=cleanup_main)
