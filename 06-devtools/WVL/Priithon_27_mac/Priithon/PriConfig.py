"""
here a list of "parameters" that control some of Priithon features are defined
They are loaded into the Y module
per "from ... import *" to prevent overwriting via reload(Y)

#old / obsolete  all parameters should start with a (single) leadig underscore
"""
__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

autoSaveSessionDir = "_py" # relative to homedir if not starting with "/"
autoSaveSessionFn  = "_pySession-%Y%m%d-autosave_%H%M%S.py" # pattern for time.strftime
# to save the list of commands into a separete file set the following != ""
# the given string will be appended to the autoSaveSessionFn (after removing the last 3 chars of that (".py")
autoSaveSessionCommands = "_commands.py" 
saveSessionDefaultPrefix = "_pySession-"
autoSaveEveryCommand = True
viewerShowFloatCoordsWhenZoomingIn = True
viewerShowComplexVals = True
defaultGfxColor = (1,0,0)             # red
viewerBkgColor = (0.2, 0.3, 0.1, 0.0) # dark greenish
viewerInterpolationMinifyDefault = 0 # for zoom<1, use linear if 1, use nearest is 0
viewerInterpolationMagnifyDefault= 0 # for zoom>1, use linear if 1, use nearest is 0
viewer2maxNumColors = 8
viewerArrowKeysShiftBy = .1  # <float> for fraction of visible area, <int> for fixed number of screen-pixels 
viewerOriginLeftBottomDefault = 1
maxOpenExceptionsWindows = 5
raiseEventHandlerExceptions = True # if False, use traceback.print_exc() instea
shellCommandTimer = "clock"
shellDisplayHookEditPopup = True
shellDisplayHookStringPopup = True
shellDisplayHookNdarrayPopup = True
shellDefaultActionOnArray = ""
shellEditEmacs = "emacsclient -nca '' +__LINE__ '__FILE__'" # Linux
# shellDefaultActionOnArray = """
# if arr.ndim==1:
#    do_ploty() 
# elif arr.ndim>1:
#    do_view()
# else:
#    do_print() 
# # or use any of : do_view2()  do_viewInLast() do_view2InLast() 
# #                 do_gridview() do_plotxy()  show_popup()
# """
email_SMTP = "mail.zedat.fu-berlin.de" # "smtp.gmail.com" # or another server -- gmail only works with TLS login ... 
email_SMTP_PORT = 587 # 0 # gmail uses port 465 (SSL) or 587 (TLS) -- 0 really means default of 25
email_SMTP_USER = '' # user account - if your SMTP server requires login
email_SMTP_PASSWORD = '' # not here please ... only if your really want to.
email_From = "" #"yourgooglemailname@gmail.com" http://lifehacker.com/111166/how-to-use-gmail-as-your-smtp-server
