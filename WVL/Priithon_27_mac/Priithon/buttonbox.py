"""
A buttonBox is a list (maybe many rows) of buttons, labels, textCtrls, checkBoxes
should be called widgetBox

Create a new buttonBox with 
>>> buttonBox(...)
it gets automatically appended to the list of all `buttonBoxes`.

You can (dynamically) add one or more controls with `buttonBoxAdd()`.
Both `buttonBoxAdd()` and `buttonBox()` take the same `itemList` argument
   each `item` in `itemList` is translated into a call of
     1) startNewRow(weight, expand)
   or
     2) addButton(label, cmd, weight, expand, tooltip, iniFcn)

if `item` is a list or tuple, up to 4 things can be specified:
     1.) `label` (i.e. text on button or text in textCtrl)
     2.) `command`  (default: None)
     3.) `weight` - items will get screen-space by relative weight (default: 1)
     4.) `expand` - adjust size vertically (on resizing buttonBox) (default: True)
     5.) `tooltip` - set mouse-over tooltip (default: the cmd string)
  .if `item` is a string, it specifies `label` and uses the above defaults for all others

`label` has the one of 3 forms (note the SPACE and the TAB):
        "C"  ,  "A\\tC"  or   "A B\\tC"
  "A" specifies the wx-control type: 'b','tb','t','c','l','sl'
      default is 'b'
      'b'  button
      'tb' togglebutton
      't'  text-control
      'c'  checkbox
      'r'  radiobutton
      'l'  static-label
      'sl' slider
      'p'  wxPanel

  "B" specifies `exec_to_name_control` - a string that is executed at creation time, 
      exec in execModule - use 'x' to refer to the wx-control, '_' to the execModule
      default is ''
  "C" specifies the wx-control"s label

  note 0) label without '\\t' defaults to control-type button (A => 'b')
  note 1) for button: if cmd is None: use label (part C) as cmd
  note 2) for textCtrl: `label` is the default value; `cmd` is triggered once initially
                        if `cmd` is '--': make read-only, set cmd to None
                        if `cmd` starts with  '**': sets wxTE_PASSWORD, set cmd to cmd[2:]
                        if `label` (part C) ends with (another) '\\t' use wxTE_MULTILINE
                        if `label` (part C) ends with '\\b' use wxTE_PASSWORD
  note 3) for checkbox: if `label` (part C) contains (another) '\\t' use ALIGN_RIGHT
  note 4) for radiobutton: if label (part C) contains (another) '\\t' use RB_GROUP (to start a new group)
  note 5) for label: command is ignored #CHECK
  note 6) for slider: label gives value,minValue,maxValue  
                        (space separated) - default: 0,0,100
  note 7) for wxPanelr: label gives width,height,[minWidth,minHeight,[maxW,maxH]]
                        (space separated) - default: -1,-1,-1,-1,-1,-1


`cmd` (`command`) will get exec'ed when the button is pressed, or the text is changed
  if will get exec'ed with globals been the buttonbox's execModule
                       and locals as follows:
   `x`: the control's value
   `_`: the execModule  -- so that you can write "_.x = x", instead of "globals()[x] = x"
   `_ev`: the wx-event (wxButtonEvent, ....)
   `_b`: the wxControl (the buttonbox's "button", or now more general, it's "gui-control") object
         ( same as _ev.GetEventObject() )


EXAMPLES:
buttonBox('print 666')
buttonBox(('devil', 'print 666'))
buttonBox([('devil', 'print 666'),
           ('xx', 'xx=99'),
          ])
buttonBox([('c\ton/off', 'print x'),
          ])


buttonBox([('l\tx-value:','',0),('t\t1234', '_.x=float(x)')])
buttonBox([('l\tx-value:','',0),('t _.myText=x\t1234', '_.textVal=x;print x')])
"""
from __future__ import absolute_import
import wx

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

try:
    buttonBoxes
except:
    buttonBoxes=[]
class _buttonBox:
    def __init__(self, title="button box",
                 execModule=None,
                 layout = "boxHoriz",
                 panel=None,
                 parent=None,
                 pos=wx.DefaultPosition,
                 size=wx.DefaultSize,
                 style=wx.DEFAULT_FRAME_STYLE):
        """
        if execModule is None: use __main__
               otherwise exec all string-command there

        layout:
          boxHoriz OR h   - horizontal BoxSizer, use "\\n" to start new row
          boxVert  OR v   - vertical   BoxSizer, use "\\n" to start new column
          (rows,cols)           - FlexGridSizer (one of nx or ny can be 0)
          (rows,cols,vgap,hgap) - FlexGridSizer (one of nx or ny can be 0)
        
        panel: put buttons into this panel
          - if None: 
             create new frame
             use title, parent, pos and style for that frame
        """
        global buttonBoxes
        self.i = len(buttonBoxes)

        if panel is None:
           # self.frame = wx.Frame(parent, -1, title + " (%d)", style=style)
           self.frame = wx.Frame(parent, -1, title, style=style,
                                 pos=pos, size=size)
           #self.frame.SetBackgroundColour(wx.SystemSettings_GetColour(wx.SYS_COLOUR_FRAMEBK))
           self.frame.SetBackgroundColour(wx.SystemSettings_GetColour(wx.SYS_COLOUR_BTNFACE))
        else:
           self.frame = panel
        self.useGridsizer = type(layout) in (list,tuple)

        if self.useGridsizer:
            #self.sizer0Vert= wx.GridSizer(*verticalLayout)
            #her Add would need pos argument -- self.sizer0Vert= wx.GridBagSizer(*verticalLayout)
            self.sizer0Vert= wx.FlexGridSizer(*layout)
        else:
            if layout[0].lower() == 'v' or \
                    len(layout)>3 and layout[3].lower() == 'v':
                self.sizer0Vert= wx.BoxSizer(wx.HORIZONTAL)
            elif layout[0].lower() == 'h' or \
                    len(layout)>3 and layout[3].lower() == 'h':
                self.sizer0Vert= wx.BoxSizer(wx.VERTICAL)
            else:
               raise ValueError("layout must be either a tuple for flexgridsizer, or contain 'v' or 'h' as 1st or 4th char (e.g. 'vert', 'boxHoriz',...)")
        #self.row=0
        self.sizers=[]
        #20110808: postpone until needed (maybe weight, expand not default) -- self.startNewRow()
        self.frame.SetSizer(self.sizer0Vert)

        #20100126 Gtk-CRITICAL **: gtk_window_resize: assertion `width > 0' failed
        #20100126 self.sizer0Vert.SetSizeHints(self.frame)
        self.frame.SetAutoLayout(1)
        #20100126 self.sizer0Vert.Fit(self.frame)

        if panel is None:
           self.frame.Show()

        #self.nButtons = 0  ## use instead: len(self.doOnEvt)
        self.doOnEvt = [] # list of event-handler lists;
        #    each is a handler called like: f(execModule, value, buttonObj, evt)

        if execModule is None:
            import __main__
            self.execModule = __main__
        else:
            self.execModule = execModule

        buttonBoxes.append(self)

        self.doOnClose = [] # (self, ev)
        self._register_onClose(self.onClose)

    def _register_onClose(self, oC):
        if self.frame.IsTopLevel():
           wx.EVT_CLOSE(self.frame, oC)
        else:
           wx.EVT_WINDOW_DESTROY(self.frame, oC)

    def onClose(self, ev):
        try:
            global buttonBoxes
            buttonBoxes[self.i] = None
        except:
            pass
        from .usefulX import _callAllEventHandlers
        _callAllEventHandlers(self.doOnClose, (self, ev), "onClose", neverRaise=True)

        if self.frame.IsTopLevel(): # 20090624
           #20090226: explicitely call close on children to make their onClose() get triggered
           # inspired by http://aspn.activestate.com/ASPN/Mail/Message/wxPython-users/2020248
           # see also: http://aspn.activestate.com/ASPN/Mail/Message/wxPython-users/2020643
           for child in self.frame.GetChildren():
                child.Close(True)

           self.frame.Destroy()
        
    def startNewRow(self, weight=1,expand=True):
        if self.useGridsizer: 
            # we use gridsizer 
            # print  "use '\\n' only with BoxSizer layouts"
            self.sizers.append( self.sizer0Vert ) # HACK 
            return

        if expand:
            expand=wx.EXPAND
        if self.sizer0Vert.GetOrientation() == wx.VERTICAL:
           ss = wx.BoxSizer(wx.HORIZONTAL)
        else:
           ss = wx.BoxSizer(wx.VERTICAL)
        self.sizers.append( ss )
        self.sizer0Vert.Add(ss, weight, expand|wx.ALL, 0)

    def addButton(self, label, cmd=None, refitFrame=True, weight=1,expand=True,tooltip=None, iniFcn=None):
        """
        if `label` is a string not containing '\\t' a button with that label is added
             and (if `cmd` is None) pushing the button will eval that same string
        if label is of form "X\\t...":
           if X is 'b'  a button will be created (that's also the default)
               cmd will be executed on button-press
               if cmd is None:
                   cmd = label

           if X is 'tb'  a toggle-button will be created
               cmd will be executed on button-press
                  (variable 'x' will contain the button status)
               if cmd is None:
                   cmd = label

           if X is 't'  a TextCtrl field will be created
               cmd will be executed on EVERY text change
                  (variable 'x' will contain the text as string)
               unless if cmd[:2] == '--':
                   this means: text field is NOT editable
               ...-part of label is the default value put in the text-field 
               #todo if ...-part of label is of form '...\\t...':
               #todo    the first part will be used as a "label" for the text field
               #todo    the second part will be the default text
               
           if X is 'c'  a CheckBox field will be created
               cmd will be executed on on click
                  (variable 'x' will contain True/False [isChecked])
               ...-part of label is a text label
                     'CheckBox' will be left-of 'text label'
                         except if ...-part contains '\\t', then
                     'CheckBox' will be right-of 'text label'
           if X is 'r'  a RadioButton field will be created
               <...>
           if X is 'l'  a StaticText label will be created
               label will be aligned to 'right'
               cmd will never be executed

            if X contains a ' ' the part after ' ' will be executed with 'x' being the control-object
               use this to keep a reference to the respective wxControl
               example: "b myButton=x"
               
        if weight is 0 button gets set to minimum size (horizontally)
           otherwise size gets distributed between all buttons
        if expand is True button expands in vertical direction with buttonBox
        if tooltip is None:
           set tooltip to be the cmd-string, or another expaining string
        else: 
           set tooltip as given
        iniFcn (string): exec like the ' '-syntax in label
               (callable): call with (execModule, button) as arguments

        NOTE:
            all execs are done in a given `execModule` as globals()
            to modify the module"s namespace the module is accessible as '_'
               e.g.: _.x = x
                     _.myTextControl = x
            (in cmd for buttons there is no 'g' - just use names directly instead: e.g. x = 5)
            `cmd` (`command`) will get exec'ed when the button is pressed, or the text is changed
              if will get exec'ed with globals been the buttonbox's execModule
                                   and locals as follows:
               `x`: the control's value
               `_`: the execModule  -- so that you can write "_.x = x", instead of "globals()[x] = x"
               `_ev`: the wx-event (wxButtonEvent, ....)
               `_b`: the wxControl (the buttonbox's "button", or now more general, it's "gui-control") object
                        ( same as _ev.GetEventObject() )
              OR if cmd is a callable: cmd(execModule, ButtonValue, buttonObj, WxEvt) is called

        """
        if '\t' in label:
            typ, label = label.split('\t',1)
        else:
            typ = 'b'

        if ' ' in typ:
            typ, exec_to_name_control = typ.split(' ', 1)
            if iniFcn is not None:
               raise ValueError("if iniFcn is given, you cannot also use exec_to_name_control syntax")
        else:
           #if iniFcn is None:
           #   exec_to_name_control = ''
           #else:
           #   exec_to_name_control = iniFcn
           exec_to_name_control = iniFcn
            

        typ = typ.lower()
        if   typ == 'b':
            b = wx.Button(self.frame, wx.ID_ANY, label)
            if cmd is None:
                cmd = label
        elif typ == 'tb':
            b = wx.ToggleButton(self.frame, wx.ID_ANY, label)
            if cmd is None:
                cmd = label
        elif typ == 'sl':
            if label:
                value,minVal,maxVal = map(int, label.split())
            else:
                value,minVal,maxVal = 0,0,100
            b = wx.Slider(self.frame, wx.ID_ANY, value,minVal,maxVal
                          #,wx.DefaultPosition, wx.DefaultSize,
                          #wx.SL_VERTICAL
                          #wx.SL_HORIZONTAL
                          #| wx.SL_AUTOTICKS | wx.SL_LABELS 
                          )

        elif typ == 'p':
           width,height=-1,-1
           minW, minH = -1,-1
           maxW, maxH = -1,-1
           if label:
              hints = map(int, label.split())
              if len(hints)>=2:
                 width,height = hints[0:2]
              if len(hints)>=4:
                 minW,minH = hints[2:4]
              if len(hints)>=6:
                 maxW,maxH = hints[4:6]

           b = wx.Panel(self.frame, wx.ID_ANY,
                         size=(width,height)
                         #,wx.DefaultPosition, wx.DefaultSize,
                         #wx.SL_VERTICAL
                         #wx.SL_HORIZONTAL
                         #| wx.SL_AUTOTICKS | wx.SL_LABELS 
                         )
           b.SetSizeHints(minW, minH, maxW, maxH)

        elif typ == 't':
            if len(label) and label[-1] =='\t':
                label = label[:-1]
                s = wx.TE_MULTILINE
            else:
                s=0
            if len(label) and label[-1] =='\b':
                label = label[:-1]
                s |= wx.TE_PASSWORD
            if type(cmd) == type("**") and len(cmd)>1 and cmd[:2] == '**':
                s |= wx.TE_PASSWORD
                cmd = cmd[2:] if len(cmd)>2 else None
                

            b = wx.TextCtrl(self.frame, wx.ID_ANY, style=s) # see below: , label)
            if type(cmd) == type("--") and len(cmd)>1 and cmd[:2] == '--':
                b.SetEditable( False )
                cmd = None
        elif typ == 'c':
            if '\t' in label:
                label, xxxx = label.split('\t',1)
                s = wx.ALIGN_RIGHT
            else:
                s=0
            b = wx.CheckBox(self.frame, wx.ID_ANY, label, style=s)
        elif typ == 'r':
            if '\t' in label:
               label, xxxx = label.split('\t',1)
               s = wx.RB_GROUP
            else:
               s=0
            b = wx.RadioButton(self.frame, wx.ID_ANY, label, style=s)
        elif typ == 'l':
            # http://lists.wxwidgets.org/archive/wx-users/msg31553.html
            # SK> Is there no way to set the vertical alignment of the label within the
            # SK> wxStaticText?
            #      No.
            # SK> If not, do any of you know of any ways to fake it?
            #      Always create the static text of the minimal suitable size (i.e. use
            # wxDefaultSize when creating it) and then pack it into a sizer using
            # spacers:
            #         wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
            #         // centre the text vertically
            #         sizer->Add(0, 1, 1);
            #         sizer->Add(text);
            #         sizer->Add(0, 1, 1);

            
            b = wx.StaticText(self.frame, wx.ID_ANY, label, style=wx.ALIGN_RIGHT)
        else:
            raise ValueError, "unknown control type (%s)"% typ
        if exec_to_name_control:
           # exec exec_to_name_control in self.execModule.__dict__, {'x':b, '_':self.execModule}
           if isinstance(exec_to_name_control, basestring):
              exec exec_to_name_control in self.execModule.__dict__, {'x':b, '_':self.execModule}
           elif callable(exec_to_name_control):
              exec_to_name_control(self.execModule, b)
           else:
              raise ValueError, "iniFcn cmd must be a string or a callable" # ?? or a list of callables"

        if tooltip is not None:
           b.SetToolTipString( tooltip )
        elif cmd:
            if isinstance(cmd, basestring):
                b.SetToolTipString( cmd )
            else:
                try:
                   b.SetToolTipString( "call '%s' [%s]" % (cmd.__name__, cmd) )
                except AttributeError:
                   b.SetToolTipString( str(cmd) )

               
        if expand:
           expand=wx.EXPAND

        try:
           ss=self.sizers[-1]
        except IndexError:
           # autostart first row, using default weight, expand
           self.startNewRow()
           ss=self.sizers[-1]

        ss.Add(b, weight, expand|wx.ALL|wx.CENTER, 0)


        ## event handling:


        if isinstance(cmd, list):
           doOnEvt = cmd
        else:
           doOnEvt = []
        
        ###################################################
        ## hand-made "templating" 
        ## a generic function to be used for each "button" type 
        ##         --- "button" can be any (here supported) wx control
        ###################################################
        ##
        ## a string with two '%s':  1) the 'x' variable 2) the "wx.EVT_BUTTON" command
        ##
        _f_template = '''
if isinstance(cmd, basestring):
   def myCmdString(selfExecMod, x, b, ev, cmd=cmd):
      exec cmd in selfExecMod.__dict__, {'_':selfExecMod, '_ev':ev, '_b':b, 'x':x}
   doOnEvt.insert(0, myCmdString)
elif callable(cmd):
   doOnEvt.insert(0, cmd)
else:
   if not (cmd is None or isinstance(cmd, list)):
       raise ValueError, "cmd must be a string or a callable or a list of callables"

def OnB(ev, self=self, b=b, iii=len(self.doOnEvt)):
    from .usefulX import _callAllEventHandlers
    _callAllEventHandlers(self.doOnEvt[iii], (self.execModule, %s, b, ev), "doOnEvt[%%d]"%%(iii,))

%s(self.frame, b.GetId(), OnB)
'''

        ##
        ###################################################
        if typ == 'b':
           exec _f_template%('ev.GetString()', 'wx.EVT_BUTTON') in globals(), locals()
        elif typ == 'tb':
           exec _f_template%('ev.GetInt()', 'wx.EVT_TOGGLEBUTTON') in globals(), locals()
        elif typ == 'sl':
           exec _f_template%('ev.GetInt()', 'wx.EVT_SLIDER') in globals(), locals()
        elif typ == 't':
           exec _f_template%('ev.GetString()', 'wx.EVT_TEXT') in globals(), locals()
        elif typ == 'c':
           exec _f_template%('ev.IsChecked()', 'wx.EVT_CHECKBOX') in globals(), locals()
           #if cmd:
           #     def OnC(ev):
           #         exec cmd in self.execModule.__dict__, {'_':self.execModule, '_ev':ev, '_b':b, 'x':ev.IsChecked()}
           #     wx.EVT_CHECKBOX(self.frame, b.GetId(), OnC)

            # TODO FIXME
            #b.SetValue(not not label) # we set "label" here so that the function is triggered already for the default value !!
            ## checkbox.SetValue:   This does not cause a wxEVT_COMMAND_CHECKBOX_CLICKED event to get emitted.
        elif typ == 'r':
           exec _f_template%('b.GetLabelText()', 'wx.EVT_RADIOBUTTON') in globals(), locals()


        if refitFrame:
            self.frame.Fit()

        self.doOnEvt.append(doOnEvt) # now even statictext (labels) have a (never used) evtHandler list 
        if typ == 't':
           b.SetValue(label) # we set "label" here so that the function is triggered already for the default value !!
        

def buttonBox(itemList=[], title="button box",
              execModule=None,
              layout = "boxHoriz",
              panel=None,
              parent=None,
              pos=wx.DefaultPosition,
              size=wx.DefaultSize,
              style=wx.DEFAULT_FRAME_STYLE,
              ret=False):
    """
    create new button box

    itemList is a list of cmd s
    `cmd` can be:
       + a string that is both button label and command to execute
       + a tuple of (label, commandString[[, weight=1[, expand=True]],tooltip=None]) (i.e.: 2,3,4 or 5 elements)

       if the string == '\\n' : that means start a new row

    title: window title (buttonBox id will be added in parenthesis)

        layout:
          boxHoriz OR h   - horizontal BoxSizer, use "\\n" to start new row
          boxVert  OR v   - vertical   BoxSizer, use "\\n" to start new column
          (rows,cols)           - FlexGridSizer (one of nx or ny can be 0)
          (rows,cols,vgap,hgap) - FlexGridSizer (one of nx or ny can be 0)
         -- all other docstring assume boxHoriz
    if execModule is None: use __main__
           otherwise exec all string-command there
    panel: put buttons into this panel
      - if None: 
         create new frame
         use title, parent, pos and style for that frame

    if `ret`: return buttonBox object
    """
    bb = _buttonBox(title, execModule, layout, panel, parent, pos, size, style)
    if len(itemList) and itemList[-1] == '\n':
       del itemList[-1]
       
    refitFrame = (size[0]==-1 or size[1]==-1)
    buttonBoxAdd(itemList, refitFrame=refitFrame) # wx.DefaultSize == (-1,-1)
    if (size[0]!=-1 or size[1]!=-1) and refitFrame: # size != wx.DefaultSize, but could also be [-1,-1]
       bb.frame.SetSize( size )

    bb.frame.Layout() # Gtk: needed if refit 
    if ret:
       return bb

def buttonBoxAdd(itemList, bb_id=-1, refitFrame=True):
    """
    add button to existing buttonBox

    itemList is a list of cmd s
    cmd can be:
       + a string that is both button label and command to execute
       + a tuple of (label, commandString [[, weight=1[, expand=True]],tooltip=None]) (i.e.: 2,3,4 or 5 elements) 

       if the string == '\n' : that means start a new row

    bb_id is the id of the buttonBox
    """
    bb = buttonBoxes[bb_id]
    if not type(itemList) in (list, tuple):
        itemList=[itemList]

    for it in itemList:
        if type(it) in (list, tuple):
            try:
                iniFcn=it[5]
            except:
                iniFcn=None
            try:
                tooltip=it[4]
            except:
                tooltip=None
            try:
                expand=int(it[3])
            except:
                expand=True
            try:
                weight = float(it[2])
            except:
                weight=1
            try:
                cmd = it[1]
            except:
                cmd = None
            label = it[0]
        else:
            label = it
            cmd=None
            weight=1
            expand=True
            tooltip=None
            iniFcn=None
        if label == '\n':
            bb.startNewRow(weight=weight,expand=expand)
        else:
            bb.addButton(label, cmd, refitFrame=False, weight=weight,expand=expand,tooltip=tooltip, iniFcn=iniFcn)

    if refitFrame:
       bb.frame.Fit()

def buttonBox_setFocus(buttonNum=0, bb_id=-1):
    """
    set a button given as "active focus" -
    hitting space or return should trigger the button
    
    buttonNum is the number of button in buttonBox (-1 is last button)

    bb_id is the id of the buttonBox
    """
    bb = buttonBoxes[bb_id]
    b = bb.frame.GetChildren()[buttonNum]
    b.SetFocus()

def buttonBox_clickButton(label, bb_id=-1):
    """
    postEvent to button with given label
    """
    bb = buttonBoxes[bb_id]
    b=wx.FindWindowByLabel(label, bb.frame)
    e=wx.CommandEvent(wx.wxEVT_COMMAND_BUTTON_CLICKED, b.GetId())
    wx.PostEvent(b, e)
