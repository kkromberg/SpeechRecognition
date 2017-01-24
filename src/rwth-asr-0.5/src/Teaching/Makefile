#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintTeaching.$(a) check$(exe)

LIBSPRINTTEACHING_O	= \
		$(OBJDIR)/BookKeeping.o     \
		$(OBJDIR)/Lexicon.o         \
	 	$(OBJDIR)/LinearSearch.o    \
		$(OBJDIR)/SearchInterface.o \
	    $(OBJDIR)/WordConditionedTreeSearch.o

CHECK_O		= $(OBJDIR)/check.o \
		  libSprintTeaching.$(a) \
		  ../Search/libSprintSearch.$(a) \
		  ../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintTeaching.$(a): $(LIBSPRINTTEACHING_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o check$(exe) $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTTEACHING_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
