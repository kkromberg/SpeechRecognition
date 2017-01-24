#!gmake
# -----------------------------------------------------------------------------
# project:    SPRINT
# -----------------------------------------------------------------------------

default:	build

TOPDIR = ..

include $(TOPDIR)/Config.make

# -----------------------------------------------------------------------------
SUBDIRS		= Am \
		  Audio \
		  Bliss \
		  Core \
		  Flow \
		  Fsa \
		  Lattice \
		  Legacy \
		  Lm \
		  Math \
		  Mc \
		  Mm \
		  Search \
		  Signal \
		  Speech \
		  Tools \
                  Teaching

# dependencies of module Speech
SPEECH_DEP = Am Flow Legacy Lm Mm Search Mc
# dependencies of module Am
AM_DEP     = Core Bliss Legacy Mm

BASIC_MODULES = Am Audio Bliss Core Flow Fsa Lattice Legacy Lm Math Mm Signal Speech Teaching
TOOLS_MODULES = $(BASIC_MODULES)


ifdef MODULE_CART
SUBDIRS         += Cart
TOOLS_MODULES   += Cart
SPEECH_DEP      += Cart
AM_DEP          += Cart
endif

ifdef MODULE_FLF_CORE
SUBDIRS         += Flf/FlfCore
BASIC_MODULES   += FlfCore
TOOLS_MODULES   += FlfCore
endif


ifdef MODULE_FLF
SUBDIRS         += Flf
TOOLS_MODULES   += Flf
endif











ifeq ($(COMPILER),icc)
endif


TARGETS		=

# -----------------------------------------------------------------------------

build:	source-version Modules.hh $(SUBDIRS)

include $(TOPDIR)/Rules.make

source-version:
	@# If a file named SourceVersion.release is available then use this for
	@# version information (for released versions of sprint).
	@# If machine chosen for compilation does not support svn or svn version
	@# is not up-to-date then discard svn version info.
	@if [ -f SourceVersion.release ]; then \
		cp SourceVersion.release SourceVersion.cc ;\
	elif svn info > /dev/null ; then \
		svn info	 > SourceVersion ;\
		svn status	>> SourceVersion ;\
		awk '{printf "\"%s\\n\"\n", $$0}' SourceVersion > SourceVersion.cc ;\
	else \
		/bin/echo -e "\"n/a\"\n" > SourceVersion.cc ;\
	fi ;

.PHONY:	source-version

modules: $(TOOLS_MODULES)

.PHONY: modules

install:
	$(MAKE) -C Tools $@

.PHONY: $(SUBDIRS)

Modules.hh : $(TOPDIR)/Modules.make
	@if [ -f $(TOPDIR)/scripts/dependencies.py ]; then \
	$(TOPDIR)/scripts/dependencies.py --basedir $(TOPDIR) --check 2> /dev/null; \
	fi
	$(ECHO) '#ifndef _MODULES_HH' > $@
	$(ECHO) '#define _MODULES_HH' >> $@
	@printf "$(foreach module, $(MODULES),'#define' $(module) '\n')" | tr -d "'" | sed -e 's/^ *//' >> $@
	$(ECHO) '#endif' >> $@


doc:
	$(MAKE) -C $@

Am: $(AM_DEP)
	$(MAKE) -C Am libSprintAm.$(a)

Audio: Flow
	$(MAKE) -C Audio libSprintAudio.$(a)

Bliss: Core
	$(MAKE) -C Bliss libSprintBliss.$(a)

Cart: Core
	$(MAKE) -C Cart libSprintCart.$(a)

Core: source-version
	$(MAKE) -C Core libSprintCore.$(a)

FFmpeg: Flow Video
	$(MAKE) -C FFmpeg libSprintFFmpeg.$(a)

Flow: Core
	$(MAKE) -C Flow libSprintFlow.$(a)

Flf: Speech
	$(MAKE) -C Flf libSprintFlf.$(a)

FlfCore : Core
	$(MAKE) -C Flf/FlfCore libSprintFlfCore.$(a)

FlfExt : Core
	$(MAKE) -C Flf/FlfExt libSprintFlfExt.$(a)

Fsa: Core
	$(MAKE) -C Fsa libSprintFsa.$(a)

Image: Flow Mm Signal Math Video Mc
	$(MAKE) -C Image libSprintImage.$(a)

Lattice: Core Fsa
	$(MAKE) -C Lattice libSprintLattice.$(a)

Legacy: Core Bliss Mm
	$(MAKE) -C Legacy libSprintLegacy.$(a)

Lm: Core Bliss
	$(MAKE) -C Lm libSprintLm.$(a)

Math: Core
	$(MAKE) -C Math

Mlp: Core Flow Math Signal
	$(MAKE) -C Mlp libSprintMlp.$(a)

Mm: Core
	$(MAKE) -C Mm libSprintMm.$(a)

Mc: Core
	$(MAKE) -C Mc libSprintMc.$(a)

Me: Core Mm Signal
	$(MAKE) -C Me libSprintMe.$(a)

OpenCV: Flow Image
	$(MAKE) -C OpenCV libSprintOpenCV.$(a)

OpenFst: Core Fsa
	$(MAKE) -C OpenFst libSprintOpenFst.$(a)

Search: Core Bliss Fsa Lattice
	$(MAKE) -C Search

Signal: Flow Math
	$(MAKE) -C Signal libSprintSignal.$(a)

Sparse: Speech
	$(MAKE) -C Sparse libSprintSparse.$(a)

Speech: $(SPEECH_DEP)
	$(MAKE) -C Speech libSprintSpeech.$(a)

Tagging: Core Fsa Me Sparse Speech Lattice Mm Bliss
	$(MAKE) -C Tagging libSprintTagging.$(a)

Teaching : Core Search
	$(MAKE) -C Teaching

Translation: Core Fsa Bliss Lm
	$(MAKE) -C Translation libSprintTranslation.$(a)
	$(MAKE) -C Translation libSprintXastur.$(a)

Tools: $(TOOLS_MODULES)
	$(MAKE) -C Tools

Video: Flow
	$(MAKE) -C Video libSprintVideo.$(a)

Vis: Core Flow
	$(MAKE) -C Vis libSprintVis.$(a)

# -----------------------------------------------------------------------------
tags:
	find . -regex '.*\.[hc][hc]?' | xargs etags

.PHONY:	tags
