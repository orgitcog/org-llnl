"""
Priithon guiParams.py
"""
from __future__ import absolute_import
import wx

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

class guiParams(object):
    def __init__(self):
        """
        guiParams is a class that can manage a set of multiple attributes -> parameters
           which can be "associated" with one (or no) or multiple gui widgets.
           The gui display and the parameter will always have consistent values !

        >>> gp = Y.guiParams()
        >>> Y.buttonBox(itemList=
        ...    gp._bbox(label="coeff0", n='c0', v=0, filter="int", slider=(0, 100), slFilter="", newLine=0)
        ...    , title="guiParams demo", verticalLayout=False, execModule=gp)

        >>> Y.buttonBox(itemList=
        ...     gp._bboxInt('int: ', 'intVal', v=0, slider=True, slmin=0, slmax=100, newLine=True)+
        ...     gp._bboxFloat('float: ', 'floatVal', v=0.0, slider=True, slmin=0.0, slmax=1.0, slDecimals=2, newLine=False)
        ...     , title="guiParams demo", verticalLayout=False, execModule=gp)
        >>> def f(v,n):
        ...     print 'n=>', v
        ...     
        >>> Y.registerEventHandler(gp._paramsDoOnValChg['floatVal'], f)
        """
        #would call  __setattr__: self.paramsVals = {}
        #would call  __setattr__: self.paramsGUIs = {}
        self.__dict__['_paramsVals'] = {}
        self.__dict__['_paramsGUIs'] = {}
        self.__dict__['_paramsDoOnValChg'] = {}
        #     # _paramsDoOnValChg keeps a list functions; get called with (val,paramName)
        self.__dict__['_paramsGUI_setAttrSrc'] = None 
        # '_paramsGUI_setAttrSrc': "short-memory" for 'which GUI triggered' 
        #  the value change - this is to prevent e.g. TxtCtrl being 'updated'
        #  while typing
        self.__dict__['_paramsOnHold'] = {} 
        # '_paramsOnHold' these are currently not triggering event handlers 
        # allowed values for `_paramsOnHold[n]` are  `True`, `False`
        # meaning: 
        #     True: changing val will not trigger evt handlers
        #     False: control just got released, trigger evt handlers, 
        #               also (first) do `del _paramsOnHold[n]`
        #               use this, to trigger evt handler even if val UNCHANGED
        self.__dict__['_simpleCounterToAutonameButtons'] = 0
        self.__dict__['_doOnPreAnyParamChanged'] = []
        self.__dict__['_doOnPostAnyParamChanged'] = []

    def __getattr__(self, n):
        try:
            v = self._paramsVals[n]
        except:
            raise AttributeError, "parameter `%s` unknown" %(n,)

        return v

    def __getitem__(self, n):
        return self.__getattr__(n)

    def __setitem__(self, n, v):
        return self.__setattr__(n, v)

    def __setattr__(self, n, v):
        """
        set/change value of given parameter with name n
        trigger all registered event handlers in _paramsDoOnValChg[n]

        call this first, before registering any gui to this parameter
           (as this sets up the necessary book keeping lists)
        """
        #self._paramsVals[n] = v

        try:
            n_to_be_held = self.__dict__['_paramsOnHold'][n]
            # True: control is down: change val, but don't trigger evt handlers
            # False: control just released, trigger evt handlers !
        except KeyError:
            n_to_be_held = None
            # None: `control`-key is and was not down, trigger only if val really changed

        try:
            # do nothing and return immediately if control is down and param did not change its value
            if n_to_be_held is None:
                vOld = self.__dict__['_paramsVals'][n]
                import numpy
                if isinstance(vOld, numpy.ndarray):
                    unchanged = False            # HACK: to prevent: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
                else:
                    try:
                        unchanged = all(vOld == v)  # iterable ?
                    except TypeError:
                        unchanged =     vOld == v       # no!
                    except ValueError:
                        unchanged = False            # HACK: list of stuff including ndarrays would still fail
                if unchanged:
                    self.__dict__['_paramsGUI_setAttrSrc'] = None
                    return
        except KeyError:
            pass

        if not n_to_be_held == True:
            try:
                vOld = self.__dict__['_paramsVals'][n]
            except KeyError:
                pass
            else:
                for f in self.__dict__['_doOnPreAnyParamChanged']:
                    f(vOld,n)

        self.__dict__['_paramsVals'][n] = v

        guis  = self.__dict__['_paramsGUIs']
        doOns  = self.__dict__['_paramsDoOnValChg']

        triggeringGUI = self.__dict__['_paramsGUI_setAttrSrc']

        if not guis.has_key(n):
            guis[n] = []
            doOns[n] = []
        else:
            def guiStuff():
                for gui in guis[n]:
                    if not gui or gui is triggeringGUI:
                        continue
                    if isinstance(gui, wx.TextCtrl):
                        #gui.SetValue( str(v) )
                        #20080821 gui.ChangeValue( str(v) )
                        gui.ChangeValue( eval(gui.val2gui+"(v)") )
                    elif isinstance(gui, wx.Button):
                        pass
                    elif isinstance(gui, wx.Panel):
                        pass  # ignore - CHECK
                    elif isinstance(gui, wx.RadioButton):
                        gui.SetValue( gui.GetLabelText() == str(v) )
                    else:
                        # wx.Slider
                        #20080821 gui.SetValue( int(v) )
                        gui.SetValue( eval(gui.val2gui+"(v)") )

            #making guiParams threadsafe !!!   CHECK !!
            if wx.Thread_IsMain():
                guiStuff()
            else:
                wx.CallAfter(guiStuff)

        self.__dict__['_paramsGUI_setAttrSrc'] = None

        if n_to_be_held == True:
            return
        if n_to_be_held == False:
            del self.__dict__['_paramsOnHold'][n]

        for f in doOns[n]:
            #try:
            f(v,n)
            #except:
            #    print >>sys.stderr, " *** error in doOnLDown **"
            #    traceback.print_exc()
            #    print >>sys.stderr, " *** error in doOnLDown **"
        for f in self.__dict__['_doOnPostAnyParamChanged']:
            f(v,n)



    def __delattr__(self, n):
        del self.__dict__['_paramsVals'][n]
        del self.__dict__['_paramsGUIs'][n]
        del self.__dict__['_paramsDoOnValChg'][n]


    # for PyCrust -- PyShell
    def _getAttributeNames(self):
        return self.__dict__['_paramsVals'].keys()

    def _getParamsGUIsTopLevelParents(self):
        """
        return a list (should be `set`) of all top level parents
        of all paramsGUIs connected with this guiParams
        """
        import operator
        q = set(map(wx.GetTopLevelParent, reduce(operator.add, self._paramsGUIs.values())))
        return list(q)
    

    #def _register(self, n, v): #, filterFnc=int):
    #    self.__dict__['_paramsVals'][n] = v
    #    self.__dict__['_paramsGUIs'][n] = []
        

    def _registerGUI(self, n, gui, val2gui=None):#=None):
        """
        connect a "new" gui control to a given paramter name

        call this only AFTER the parameter 'n' has been set -- see __setattr__
        """
        try:
            l=self.__dict__['_paramsGUIs'][n]
        except KeyError:
            raise AttributeError, "parameter `%s` unknown" %(n,)            
            #    l=self.__dict__['_paramsGUIs'][n] = []

        if val2gui is not None:
            gui.val2gui = val2gui
        elif isinstance(gui, wx.TextCtrl):
            gui.val2gui = "str" 
        elif isinstance(gui, wx.Slider):
            gui.val2gui = "int" 
        elif isinstance(gui, (wx.CheckBox, wx.ToggleButton)):
            gui.val2gui = "bool" 
        elif isinstance(gui, wx.RadioButton):
            gui.val2gui = "<unused>" 
        elif isinstance(gui, wx.Panel):
            gui.val2gui = "<unused>" 
        elif isinstance(gui, wx.Button):
            gui.val2gui = None
        else:
            print "WARNING: GuiParams: what is the type of this gui:", gui
            gui.val2gui = "int" 
            
        #if gui in not None:
        l.append(gui)

    def _unregisterGUI(self, n, gui):
        try:
            l=self.__dict__['_paramsGUIs'][n]
        except KeyError:
            raise AttributeError, "parameter `%s` unknown" %(n,)
        l.remove(gui)


    def _setValueAndTriggerHandlers(self, n,v):
        """
        sets variable `n` to value `v`
        use this to force trigger events even if `value` is equal to current value of `n`
        """
        self.__dict__['_paramsOnHold'][n] = False
        self.__setattr__(n, v)

    def _setValueWithoutTriggerOrGUIupdate(self, n,v):
        """
        sets variable `n` to value `v`
        use this to circumvent any trigger events 
        and not updating any of the connected GUIs

        self.__dict__['_paramsVals'][n] = v
        """
        self.__dict__['_paramsVals'][n] = v

    def _holdParamEvents(self, n=None, hold=True):
        """
        if `hold`, changing parameter `n` will _not_ trigger the event handlers being called
        otherwise, reset to normal, next change will trigger

        if `n` is None:
           apply / reset hold for all params
        """
        if n is None:
            for n in self.__dict__['_paramsVals'].iterkeys():
                self._holdParamEvents(n, hold)
            return
        if isinstance(n, (list, tuple)):
            for nn in n:
                self._holdParamEvents(n, hold)
            return


        if hold:
            self.__dict__['_paramsOnHold'][n]=True
        else:
            try:
                del self.__dict__['_paramsOnHold'][n]
            except KeyError:
                pass
    def _enableParamGUIs(self, n=None, enable=True):
        """
        enable or diable all guis for param n.
        `n`: name of gui param, or a list/tuple thereof

        if `n` is None:
           enable/diablea all params
        """
        if n is None:
            for n in self.__dict__['_paramsVals'].iterkeys():
                self._enableParamGUIs(n, enable)
            return
        if isinstance(n, (list, tuple)):
            for nn in n:
                self._enableParamGUIs(nn, enable)
            return

        for g in self._paramsGUIs[n]:
            g.Enable(enable)

            
    def _spiffupCtrl(self, b, n, arrowKeyStep):
#                          evt.ControlDown(), 'C'),
#                         (evt.AltDown(),     'A'),
#                         (evt.ShiftDown(),   'S'),
#                         (evt.MetaDown(),
        """
        make control respond to keys:
        arrow up/down change value
        with Shift values change 10-fold faster
        with Ctrl being pressed, event handler are not getting called 
        """
        def OnKeyUp(evt):
           keycode = evt.GetKeyCode()
           if keycode == wx.WXK_CONTROL:
               try:
                   self.__dict__['_paramsOnHold'][n]=False
                   v = self.__dict__['_paramsVals'][n]
                   self.__setattr__(n,v)
               except KeyError:
                   pass
           evt.Skip()

        def OnKeyDown(evt):
           keycode = evt.GetKeyCode()

           if keycode == wx.WXK_CONTROL:
               self.__dict__['_paramsOnHold'][n]=True

           if evt.ShiftDown():
               arrowKeyStepLocalVar=arrowKeyStep*10
           else:
               arrowKeyStepLocalVar=arrowKeyStep
           if keycode == wx.WXK_UP:
               v = self.__dict__['_paramsVals'][n]+arrowKeyStepLocalVar
               self.__setattr__(n,v)
           elif keycode == wx.WXK_DOWN:
               v = self.__dict__['_paramsVals'][n]-arrowKeyStepLocalVar
               self.__setattr__(n,v)
           else:
               evt.Skip()

        b.Bind(wx.EVT_KEY_DOWN, OnKeyDown)
        b.Bind(wx.EVT_KEY_UP, OnKeyUp)
           


#     def _bbox(self, label, n, v, filter='int', slider=(0,100), slFilter='', newLine=False):
#         '''
#         return list of tuple to be used in buttonBox contructors itemList

#         label: static text label used in button box (e.g. "min: ")
#         n:     guiParms var name (as string!)
#         v:     inital value
#         filter: string, e.g. 'int' ; function to convert text field's value into guiParam value
#         slider: tuple of min,max value for slider ; 
#                 None if you don't want a slider
#         slFilter: string, e.g. '' ; function to convert slider value into guiParam value: TODO FIXME!!
#         '''
#         self.__setattr__(n,v)
#         l= [
#             ("l\t%s\t"%label,   '', 0,0),
#             ("t _._registerGUI('%s', x)\t%s"%(n, self.__dict__['_paramsVals'][n]), "_.%s = %s(x)"%(n,filter), 0,0),
#             ]
#         if slider is not None:
#             slMin, slMax = slider
#             l += [
#             ("sl _._registerGUI('%s', x)\t%d %d %d"%(n, self.__dict__['_paramsVals'][n], slMin, slMax), "_.%s = %s(x)"%(n,slFilter), 1,0),
#             ]
        
#         if newLine:
#             l += ['\n']
#         return l

    def _bboxNewline(self, weight=0, expand=True):
        """
        return list of tuple to be used in buttonBox contructors itemList
        
        shortcut for [ ('\n', '', weight, expand) ]
        """
        return [ ('\n', '', weight, expand) ]

    def _bboxButton(self, label, n=None, v=0, regFcn=None, regFcnName=None, weight=1, expand=True,
                    newLine=False, tooltip=""):
        """
        return list of tuple to be used in buttonBox contructors itemList
        
        guiParam buttons are counters, incremented by 1 one each click

        label: static text label used in button box (e.g. "pushMe!")
        n:     guiParms var name (as string!) 
               -- None means: generate auto-name: 'button1', 'button2', ...
        v:     inital value

        """
        if n is None:
            self.__dict__['_simpleCounterToAutonameButtons'] += 1
            ccc = self.__dict__['_simpleCounterToAutonameButtons']
            n = 'button%d'%(ccc,)
            #self.__dict__['_simpleCounterToAutonameButtons']

        def fcn(execModule, value, buttonObj, evt):
            #print execModule, value, buttonObj, evt
            #print '-----------------------'

            #not needed for wxButton: self.__dict__['_paramsGUI_setAttrSrc'] = buttonObj
            self.__setattr__(n, self.__getattr__(n)+1)
        self.__setattr__(n,v)

        def iniFcn(exeMod, x):
            self._registerGUI(n, x)

        t = "b\t%s"%(label,)
        l = [ (t, fcn, weight,expand,tooltip, iniFcn) ]
        if newLine:
            l += ['\n']

        # register event handlers
        if regFcn is not None:
            from .usefulX import registerEventHandler
            registerEventHandler(self.__dict__['_paramsDoOnValChg'][n], 
                                  newFcn=regFcn, newFcnName=regFcnName) #, oldFcnName='', delAll=False)

        return l

    def _bboxPanel(self, n, weight=1, expand=True,
                   width=-1, height=-1,
                   sizer="h",
                   newLine=False, tooltip=""):
        """
        return list of tuple to be used in buttonBox contructors itemList
        
        create a wxPanel, `n` refers wxPython object

        n:     guiParms var name (as string!) 
        sizer: creates and sets BoxSizer: 
                   "h..." or "v..." for horizontal or vertical
                   a given wxSizer object is used as sizer
                   otherwise no sizer will be set
        """
        # def fcn(execModule, value, buttonObj, evt):
        #     #print execModule, value, buttonObj, evt
        #     #print '-----------------------'

        #     #not needed for wxButton: self.__dict__['_paramsGUI_setAttrSrc'] = buttonObj
        #     self.__setattr__(n, self.__getattr__(n)+1)

        def iniFcn(exeMod, x):
            v = x
            self.__setattr__(n,v)
            self._registerGUI(n, x)
            if sizer:
                if isinstance(sizer, basestring):
                    bsOrient =  sizer.lower()[0]
                    if bsOrient == "h":
                        ss = wx.BoxSizer(wx.HORIZONTAL)
                    elif bsOrient == "v":
                        ss = wx.BoxSizer(wx.VERTICAL)
                    else:
                        raise ValueError("sizer must be string starting with 'h' or 'v', \n"
                                         "or sizer object or nothing")
                else:
                    ss = sizer
                v.SetSizer(ss)
                wx.CallAfter(v.Layout)  # CHECK

        sizes = ' '.join(map(str,[width, height, width, height]))
        t = "p\t%s"%(sizes,)
        fcn = ''
        l = [ (t, fcn, weight,expand,tooltip, iniFcn) ]
        if newLine:
            l += ['\n']

        # # register event handlers
        # if regFcn is not None:
        #     from .usefulX import registerEventHandler
        #     registerEventHandler(self.__dict__['_paramsDoOnValChg'][n], 
        #                           newFcn=regFcn, newFcnName=regFcnName) #, oldFcnName='', delAll=False)

        return l

    def _bboxInt(self, label, n, v=0, 
                 slider=True, slmin=0, slmax=100, newLine=True,
                 val2txt="str",
                 labelWeight=0, labelExpand=False, 
                 textWeight=0, textExpand=False, textWidth=-1,
                 sliderWeight=1, sliderExpand=False, sliderWidth=100,
                 tooltip="", regFcn=None, regFcnName=None):
        """
        val2txt: can somthing like: val2txt="'%03d'%", because it gets prepended before "(x)"
        """
        return self._bbox_genericTextAndSlider(label, n, v, 
                                    txt2val=int, val2txt=val2txt,
                                    slider=(slmin,slmax) if slider else None, 
                                    sl2val=int, 
                                    val2sl='int', 
                                    arrowKeyStep=1,
                                    newLine=newLine,
                                    labelWeight=labelWeight, labelExpand=labelExpand,
                                    textWeight=textWeight, textExpand=textExpand, textWidth=textWidth,
                                    sliderWeight=sliderWeight, sliderExpand=sliderExpand, sliderWidth=sliderWidth,
                                    tooltip=tooltip, regFcn=regFcn, regFcnName=regFcnName)
    def _bboxFloat(self, label, n, v=0.0, 
                   slider=True, slmin=0.0, slmax=1.0, slDecimals=2, 
                   newLine=True,
                   val2txt="str",
                   labelWeight=0, labelExpand=False, 
                   textWeight=0, textExpand=False, textWidth=-1,
                   sliderWeight=1, sliderExpand=False, sliderWidth=100,
                   tooltip="", regFcn=None, regFcnName=None):
        """
        val2txt: can somthing like: val2txt="'%.2f'%", because it gets prepended before "(x)"
        """
        return self._bbox_genericTextAndSlider(label, n, v, 
                                    txt2val=float, val2txt=val2txt,
                                    slider=(slmin,slmax)  if slider else None, 
                                    sl2val=(lambda x:x/10**slDecimals), 
                                    val2sl='(lambda x:x*%f)'%(10**slDecimals,), 
                                    arrowKeyStep=.1**slDecimals,
                                    newLine=newLine,
                                    labelWeight=labelWeight, labelExpand=labelExpand, 
                                    textWeight=textWeight, textExpand=textExpand, textWidth=textWidth,
                                    sliderWeight=sliderWeight, sliderExpand=sliderExpand, sliderWidth=sliderWidth,
                                    tooltip=tooltip, regFcn=regFcn, regFcnName=regFcnName)
    def _bboxText(self, label, n, v="", newLine=True,
                  labelWeight=0, labelExpand=False, 
                  textWeight=1, textExpand=False, textWidth=-1,
                  textMultiline=False, textPassword=False, 
                  tooltip="", regFcn=None, regFcnName=None):
        return self._bbox_genericTextAndSlider(label, n, v, txt2val=str, 
                                    slider=None, 
                                    arrowKeyStep=0,
                                    newLine=newLine,
                                    labelWeight=labelWeight, labelExpand=labelExpand, 
                                    textWeight=textWeight, textExpand=textExpand, textWidth=textWidth,
                                    textMultiline=textMultiline, textPassword=textPassword, 
                                    tooltip=tooltip, regFcn=regFcn, regFcnName=regFcnName)
    

    def _bboxChoice(self, label, n, itemList, i=0, newLine=False,
                      txt2val=str,
                      tooltip="", regFcn=None, regFcnName=None):
        """
        return list of tuple to be used in buttonBox contructors itemList

        label: static text label used in button box (e.g. "min: ")
        n:     guiParms var name (as string!)
        itemList: list of labels of each radio button -- labels can be anything that str() understands
        i:     inital value - its index in itemList
        txt2val: function to be used to convert radiobutton labels to value, e.g. str or int

        tooltip can be a list: then each item gets corresponding tooltip
        """
        def fcn(execModule, value, buttonObj, evt):
            self.__dict__['_paramsGUI_setAttrSrc'] = buttonObj
            self.__setattr__(n, txt2val(value))
        self.__setattr__(n, itemList[i])
 
        l=[]
        weight = 0
        expand = False
        if label:
            tooltip1 = tooltip if isinstance(tooltip, basestring) else ""
            l.append( ("l\t%s"%label,   '', weight,expand,tooltip1) )

        #controls = controls.split()
        for ii,item in enumerate(itemList):
            def iniFcn(exeMod, x):
                x.SetValue(ii==i)
                self._registerGUI(n, x)

            tooltip1 = tooltip if isinstance(tooltip, basestring) else tooltip[ii]
            t = "r\t%s"%(item, )
            if ii==0: # if c[-1] in '-n': # start new group
                t += '\t'
            l.append( (t, fcn, weight,expand,tooltip1, iniFcn) )
        if newLine:
            l += ['\n']

        # register event handlers
        if regFcn is not None:
            from .usefulX import registerEventHandler
            registerEventHandler(self.__dict__['_paramsDoOnValChg'][n], newFcn=regFcn, newFcnName=regFcnName) #, oldFcnName='', delAll=False)

        return l

    def _bboxBool(self, label, n, v=False, controls='cb', newLine=False,
                  tooltip="", regFcn=None, regFcnName=None):
        """
        return list of tuple to be used in buttonBox contructors itemList

        label: static text label used in button box (e.g. "min: ")
        n:     guiParms var name (as string!)
        v:     inital value

        controls: string of space spearated "codes" specifying what wxControls should be shown
                 (only first (and maybe last) case-insensitive char is significant)
            "l"  - text label
            "tb" - toggle button
            "c"  - checkbox -- append an "r" make it right-aligned ("cb","cbL","cbR","cR" all match this one...)
          if this code is followed by one (int) number (space separated), 
             its value is used as "weight"
          if this is followed by another number (space separated),
             its value is used as "expand" (bool)
        """
        def cmdFcn(execModule, value, buttonObj, evt):
            self.__dict__['_paramsGUI_setAttrSrc'] = buttonObj
            self.__setattr__(n,bool(value))
        self.__setattr__(n,v)

        def iniFcn(exeMod, x):
            x.SetValue(v)
            self._registerGUI(n, x)

        l=[]
        controls = controls.split()
        for i,c in enumerate(controls):
            try:
                int(c)
                continue
            except ValueError:
                pass
            c=c.lower()
            try:
                weight = int(controls[i+1])
            except:
                weight = 0
            try:
                expand = bool(int(controls[i+2]))
            except:
                expand = False
                
            if    c[0] == 'l':
                l.append( ("l\t%s\t"%label,   '', weight,expand,tooltip) )
            elif  c[0] == 't':
                t = "tb\t%s"%(label,)
                l.append( (t, cmdFcn, weight,expand,tooltip, iniFcn) )
            elif  c[0] == 'c':
                t = "c\t%s"%(label,)
                if c[-1] == 'r': # right aligned
                    t += '\t'
                l.append( (t, cmdFcn, weight,expand,tooltip, iniFcn) )
            else:
                raise ValueError, "bool control type '%s' not recognized"%(c,)
        if newLine:
            l += ['\n']

        # register event handlers
        if regFcn is not None:
            from .usefulX import registerEventHandler
            registerEventHandler(self.__dict__['_paramsDoOnValChg'][n], newFcn=regFcn, newFcnName=regFcnName) #, oldFcnName='', delAll=False)

        return l

    def _bbox_genericTextAndSlider(self, label, n, v=.5, txt2val=float, val2txt="str",
                        slider=(0,1), sl2val=(lambda x:x/100), val2sl='(lambda x:x*100)', 
                        arrowKeyStep=0.01, newLine=False,
                        labelWeight=0, labelExpand=False, textWeight=0, 
                        textMultiline=False, textPassword=False, 
                        textExpand=False, textWidth=-1, 
                        sliderWeight=1, sliderExpand=False, sliderWidth=100,
                        tooltip="", regFcn=None, regFcnName=None):
        """
        return list of tuple to be used in buttonBox contructors itemList

        label: static text label used in button box (e.g. "min: ")
        n:     guiParms var name (as string!)
        v:     inital value
        txt2val: function to convert text field's value into guiParam value
        slider: tuple of min,max value for slider ; 
                None if you don't want a slider
        sl2val: function to convert slider value into guiParam value:
        """
        def fcnTxt(execModule, value, buttonObj, evt):
            #20110826 if len(value):
            self.__dict__['_paramsGUI_setAttrSrc'] = buttonObj
            try:
                v = txt2val(value)
            except:
                pass
            else:
                self.__setattr__(n, v)
        def fcnSlider(execModule, value, buttonObj, evt):
            self.__dict__['_paramsGUI_setAttrSrc'] = buttonObj
            self.__setattr__(n, sl2val(value))

        self.__setattr__(n,v)

        def iniFcnTxt(exeMod, x):
            if arrowKeyStep:
                self._spiffupCtrl(x,n, arrowKeyStep)
            if textWidth>=0:
                x.SetSizeHints(textWidth,-1)
            self._registerGUI(n, x, val2gui=val2txt)

        def iniFcnSlider(exeMod, x):
            if arrowKeyStep:
                self._spiffupCtrl(x,n, arrowKeyStep)
            if sliderWidth>=0:
                x.SetSizeHints(sliderWidth,-1)
            self._registerGUI(n, x, val2gui=val2sl)

        if label:
            l= [
                ("l\t%s\t"%label,   '', labelWeight,labelExpand, tooltip),
                ]
        else:
            l=[]

        lab = ""
        if textMultiline:
            lab += '\t'
        if textPassword:
            lab += '\b'
        l += [
            ("t\t%s%s"%(self.__dict__['_paramsVals'][n], lab),
             fcnTxt, textWeight,textExpand, tooltip, iniFcnTxt),
            ]


        if slider is not None:
            slMin, slMax = [eval(val2sl)(v) for v in slider]
            slVal0 = eval(val2sl)(self.__dict__['_paramsVals'][n])
            l += [
            ("sl\t%d %d %d"%(slVal0, slMin, slMax),
             fcnSlider, 
             sliderWeight,sliderExpand, tooltip, iniFcnSlider),
            ]
        
        if newLine:
            l += ['\n']

        # register event handlers
        if regFcn is not None:
            from .usefulX import registerEventHandler
            registerEventHandler(self.__dict__['_paramsDoOnValChg'][n], 
                                  newFcn=regFcn, newFcnName=regFcnName) #, oldFcnName='', delAll=False)

        return l


    def _clearDeletedGuis(self):
        for n,v in self._paramsGUIs.iteritems():
            self._paramsGUIs[n] = filter(None, v)

    def _guiBox(self, itemList=[], title="gui parameters",
                layout = "boxHoriz",
                panel=None,
                parent=None,
                pos=wx.DefaultPosition,
                size=wx.DefaultSize,
                style=wx.DEFAULT_FRAME_STYLE,
                execModule=None,
                ret=False):
        """
        build a GUI interface for these parameters

        shortcut for `Y.buttonBox( ... )`

        if execModule is None: use __main__
           otherwise exec all string-command there
        """
        from .buttonbox import buttonBox
        from .usefulX import registerEventHandler
        bb=buttonBox(itemList=itemList, title=title,
                     execModule=execModule,
                     layout=layout, panel=panel, parent=parent, pos=pos, size=size, style=style, ret=True)
        

        # install new close handler who clears deleted GUIs _AFTER_ all other doOnClose handlers were called
        import new
        bb.onClose_orig_buttonBox = bb.onClose
        def onClose_clearGuiParmsGui(other_self, ev):
            other_self.onClose_orig_buttonBox(ev)

            #doesn't work (tested on Gtk-Linux):
            #wx.CallAfter(self._clearDeletedGuis) # handler in onClose must go first...

            # so we do this "by hand":
            if bb.frame.IsTopLevel():
                from .usefulX import iterChildrenTree
                chds = [w for w in iterChildrenTree(bb.frame)]
                for n,v in self._paramsGUIs.iteritems():
                    self._paramsGUIs[n] = [vv for vv in v if vv not in chds]

        bb.onClose = new.instancemethod(onClose_clearGuiParmsGui, bb,bb.__class__)
        bb._register_onClose(bb.onClose)

        import weakref
        bb.gp = weakref.proxy(self) # in case gp would get lost


        if ret:
            return bb

# class guiHistValue:
#     def __init__(self, id=-1, leftOrRight=0):
#         """
#         leftOrRight =0: use left brace
#         leftOrRight =1: use right brace
#         """
#         self.id = id
#         self.leftOrRight = leftOrRight
         

#     def SetValue(self, v):
#         from .all import Y
#         if self.leftOrRight ==0:
#             Y.vHistScale(id=self.id, amin=v, amax=None, autoscale=False)
#         else:
#             Y.vHistScale(id=self.id, amin=None, amax=v, autoscale=False)
