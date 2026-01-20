# -*- Makefile -*-
#__author__  = "Sebastian Haase <haase@msg.ucsf.edu>"
#__license__ = "BSD license - see LICENSE file"

#ifneq (,$(findstring CYGWIN,$(shell uname)))

include $(PRCOMMON)/priithon_$(shell uname).mak

#-include means: include - BUT just ignore if file doesn't exist
-include Makefile_$(shell uname).dep
