"""
some help strings used in help menu
"""
from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

def PRE(txt):
    """return "<pre>" + txt + "</pre>"
    special treatment for "<", ">", "&"
    """
    import re
    #broken wxHtml txt = re.sub('<', '&lt;', txt)
    #broken wxHtml txt = re.sub('>', '&gt;', txt)
    #txt = re.sub('<', '</pre><tt>&lt;</tt><pre>', txt)
    txt = re.sub('<', '>', txt) # BAD FIX HACK
    #txt = re.sub('&', '&amp;', txt)
    return "<pre>" + txt + "</pre>"

#import sys
#pyShell_helpText = sys.app.frame.shell.helpText
from .py.shell import helpText as pyShell_helpText

priShell_helpHTML = """
"<a href='google.com/search?q=priithon'>google search priithon</a>"

<h3>Keyboard Interrupt</h3>
You can interrupt long-running commands by pressing <b>Ctrl-C</b> into the <i>terminal</i> (that is, not the Priithon shell, but the "black" Unix/command window in the background) 
<h3>Tab completion</h3>
Pressing &lt;TAB&gt; completes variable, function, class and module names. <br>
It also completes file and directory names when &lt;TAB&gt; is pressed after an opening single or double quote; if a * (star) is between the current cursor positin and the opening quote <em>only</em> files or directories matching that pattern are shown.<br>
&lt;TAB&gt; also (re-)shows the function tool-tip, and completes argument options.

<h3>Priithon magic</h3>
Start a line with a SPACE to have it executed as a system command
 - this uses <tt>U.exe( ... )</tt><br>
example: <tt>  >>> &nbsp; &nbsp;  ls -l</tt>
Trailing  <tt>#</tt> or <tt>;</tt> suppresses output (<tt>_</tt> is set to output)
<h3>Shell keyboard shortcuts</h3>
""" \
+ PRE(pyShell_helpText)

viewerHelp ="""
<h3>key commands</h3>
""" +\
PRE("""\
`*` on NumPad  -> auto histogram scaling
`o` cycle through 4 /origin/ modes:
     (0,0) at left bottom
     (0,0) at left top
     (0,0) at center ( for fft images )
     (0,0) at center &amp; "double width" ( for rfft images)
`c` cycle through many /color map/ modes:
     gray scale
     gray scale /logarithmic/
     rainbow red to blue
     /circular/ rainbow - good for phase images, where -PI is the same as +PI
     "heat like" black body
     fast cycling rainbow with 0 black - good to see iso-conturs and gradients
`C` open popup window for setting gamma value
`a` auto-adjust frame size to image as displayed
`p` toggle /phases/ and /amplitudes/ display for complex image data
`f` open new viewer showing the ("half-shaped",real-) fft(2d) of the image
`F` open new viewer showing the inverse fft(2d) of the current image (should be a "half-shaped" fft image)
`v` open new viewer showing "maximum intensity projection" along z-axis
`V` open new viewer showing "mean intensity projection" along z-axis
`g` cycle through: no /grid/, one pixel grid, ten pixel grid
`x` open new viewer showing x-z side on view
`y` open new viewer showing y-z side on view
`0` reset zoom to one pixel per data point and move image to left bottom in viewer frame
`9` center image in viewer frame d "double" zoom - zoom in
`h` "half" zoom - zoom out
`d` "double" zoom - zoom in
<page up/down> zoom in/out 
`l` toggle histogram with and with out /logarithmic/ /y/-axis
<Home>
arrow keys left,right - walk through /z/ axis (or what ever the higher dimensions are for)
arrow keys up, down - walk through /t/ axis (or what ever the higher dimensions are for)
arrow keys with <shift> AND <control> - shift image by quarter of it's size in the respective direction
arrow keys with ONLY <control> - shift image by a configurable amount in the respective direction
""")+"""\

<h3>Mouse interaction in image part</h3>
""" + PRE("""\
press middle mouse button to drag image
press middle mouse button with <shift> or <Ctrl> key to zoom (move mouse up/down)
 or use mouse wheel to zoom in/out
right mouse button gives "context menu"
""") + """\

<h3>Mouse interaction in histogram part</h3>
""" + PRE("""\
press middle mouse button to move histogram graph:
   left/right; up/down zooms in/out
   also you can use the mouse wheel to zoom
press left mouse button to change image "scaling" (meaning: it brightness and contrast):
   click close to left (red) bracket to change min (black) value,
   click close to right red bracket to modify the max (white) value
   or click in the center to move them both together.
right mouse button gives "context menu"
""") + """\
<h3>shell commands</h3>
""" + PRE("""\
Y.vd(): return data array of whats displayed in a viewer
...
""") + """\
"""

def _helpAbout():
    global helpAbout
    import sys, wx

    try:
        import os, glob, stat, operator, time
        from . import useful as U
        d = os.path.dirname(U.__file__)
        ff = glob.glob(os.path.join(d, '*.py'))
        ffdates  = [(f,os.stat(os.path.join(d, f))[stat.ST_MTIME]) for f in ff]
        ffdates.sort(key=operator.itemgetter(1), reverse=True)
        pyfn   = [os.path.basename(f[0]) for f in ffdates[:3]]
        pydate = [time.ctime(f[1]) for f in ffdates[:3]]
    except:  # in case files are not found 
        pyfn = ['???']*3
        pydate = ['???']*3
    try:
        import os # , glob, stat, operator, time
        from . import useful as U
        d = os.path.dirname(U.__file__)
        priDir = d
    except:
        priDir = '???'

    text = """
Priithon is a open source platform 
for multi dimensional image analysis 
and algorithm development. 

Priithon is a collection of many other open source projects.
Most Priithon-specific code has been 
written by Sebastian Haase.

Priithon is hosted at
http://code.google.com/p/priithon

The wx python py-shell was originally written by Patrick K. O\'Brien

"""+ (      'Priithon version: the 3 newest py files are\n' +
            '\t\'%s\' dated %s\n'%(pyfn[0],pydate[0]) +
            '\t\'%s\' dated %s\n'%(pyfn[1],pydate[1]) +
            '\t\'%s\' dated %s\n'%(pyfn[2],pydate[2]) +
            'Priithon package base dir:\n\t\'%s\'\n'%(priDir,) +
            '\n' +
            'Platform: %s\n' % sys.platform +
            'Python Version: %s\n' % sys.version.split()[0] +
            'wxPython Version: %s\n' % wx.VERSION_STRING + 
            '\t(%s)\n' % ", ".join(wx.PlatformInfo[1:])
            )
    helpAbout = text

_helpAbout()
