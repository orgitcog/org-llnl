CC= mpicc

# Compiler flags
CFLAGS= -I../include
#OPT= -g -O0 -Wall
OPT= -Wall -O3 -finline-limit=1000000

SHELL= /bin/sh

# Target library name
TARGET= librecbis.a

SRCS= \
  alltoall_sparse.c approx_median.c     balance.c  balance_hist.c  \
  equalize.c        exchange.c          kstat.c    parkstat.c      \
  rectimer.c        redistribute.c      treecomm.c

# Create names of object files
OBJS= $(addsuffix .o, $(basename $(SRCS)))

# Rule to compile C source file into object file
%.o: %.c
	$(CC) $(CFLAGS) $(OPT) -o $@ -c $<
# Create dependency file from C source
%.d: %.c
	(echo -n "$@ " ; $(CC) $(CFLAGS) -MM $< ) > $@

# Default rule
all: target_dep target_rule

# Clean objects, dependency files, and target library
clean:
	rm -f $(OBJS) $(OBJS:.o=.d) $(TARGET)

# Install (copy) target library into ../lib/
#install: $(TARGET)
#	mkdir -p ../lib
#	cp -a $(TARGET) ../lib/

# Create names of depenency files
target_dep: $(addsuffix .d, $(basename $(SRCS)))

# Link library
target_rule: $(OBJS)
	rm -rf $(TARGET)
	ar rc $(TARGET) $(OBJS)

# Include dependency files
ifneq ($(MAKECMDGOALS),clean)
-include $(OBJS:.o=.d)
endif
