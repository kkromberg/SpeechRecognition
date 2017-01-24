#!make

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= lattice-processor$(exe)

LATTICE_PROCESSOR_O = $(OBJDIR)/LatticeProcessor.o \
		../../Speech/libSprintSpeech.$(a) \
		../../Lattice/libSprintLattice.$(a) \
		../../Lm/libSprintLm.$(a) \
		../../Am/libSprintAm.$(a) \
		../../Mm/libSprintMm.$(a) \
		../../Mc/libSprintMc.$(a) \
		../../Legacy/libSprintLegacy.$(a) \
		../../Bliss/libSprintBliss.$(a) \
		../../Audio/libSprintAudio.$(a) \
		../../Search/libSprintSearch.$(a) \
		../../Signal/libSprintSignal.$(a) \
		../../Flow/libSprintFlow.$(a) \
		../../Math/libSprintMath.$(a) \
		../../Math/Lapack/libSprintMathLapack.$(a) \
		../../Core/libSprintCore.$(a) \
		../../Fsa/libSprintFsa.$(a) \
		../../Flf/FlfCore/libSprintFlfCore.$(a)

ifdef MODULE_CART
LATTICE_PROCESSOR_O += ../../Cart/libSprintCart.$(a)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

lattice-processor$(exe): $(LATTICE_PROCESSOR_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(LATTICE_PROCESSOR_O)))
