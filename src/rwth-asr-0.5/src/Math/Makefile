#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	= Lapack
TARGETS		= libSprintMath.$(a) check$(exe)

LIBMATH_O	= \
		$(OBJDIR)/AcousticalAnalyticFunctions.o \
		$(OBJDIR)/AnalyticFunctionFactory.o \
		$(OBJDIR)/EigenvalueProblem.o \
		$(OBJDIR)/FastFourierTransform.o \
		$(OBJDIR)/Module.o \
		$(OBJDIR)/PiecewiseLinearFunction.o





CHECK_O		= $(OBJDIR)/check.o \
		libSprintMath.$(a) \
		Lapack/libSprintMathLapack.$(a) \
		Nr/libSprintMathNr.$(a) \
	        ../Core/libSprintCore.$(a)


# -----------------------------------------------------------------------------

default: build

build: libSprintMath.$(a) $(SUBDIRS)

all: $(TARGETS)

.PHONY: $(SUBDIRS)

Nr:
	$(MAKE) -C $@ libSprintMathNr.$(a)
Lapack:
	$(MAKE) -C $@ libSprintMathLapack.$(a)

libSprintMath.$(a): $(LIBMATH_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $^ -o check$(exe) $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBMATH_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
