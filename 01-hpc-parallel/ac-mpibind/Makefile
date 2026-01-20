TARGET   = libmpibind.so

CC       = gcc
LINKER   = gcc

HWLOC_ROOT = /g/g90/loussert/SOFTS/hwloc-2.0.4/INSTALL/
HWLOC_INCS = -I$(HWLOC_ROOT)/include
HWLOC_LIBS = -L$(HWLOC_ROOT)/lib -lhwloc

CFLAGS   = -Wall $(HWLOC_INCS) -I. -fpic
LFLAGS   = -Wall $(HWLOC_LIBS) -I. -shared

SRCDIR   = core/src
INCDIR	 = core/include
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.c)
INCLUDES := $(wildcard $(INCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

rm       = rm -f


$(BINDIR)/$(TARGET): $(OBJECTS)
	@$(LINKER) $(OBJECTS) $(LFLAGS) -o $@
	@echo "Linking complete!"

$(OBJECTS): $(OBJDIR)/%.o : $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiled "$<" successfully!"

.PHONY: clean
clean:
	@$(rm) $(OBJECTS)
	@echo "Cleanup complete!"

.PHONY: cleaner 
cleaner: clean
	@$(rm) $(BINDIR)/$(TARGET)
	@echo "Executable removed!"
