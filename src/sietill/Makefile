#########################################################################
#                                                                       #
#     COPYRIGHT (C) 2015 Lehrstuhl fuer Informatik VI, RWTH Aachen      #
#                                                                       #
#########################################################################

#########################################################################
# set dependency file:                                                  #
#########################################################################
INSTDIR = $(shell pwd)
LOGFILE =

#########################################################################
# define targets:                                                       #
#########################################################################
TARGETS		= sietill

#########################################################################
# compiler options                                                      #
#########################################################################
CXX       = g++
CXXFLAGS += -Wall -O2 --std=c++11 -pedantic -msse -msse2 -msse3
DFLAGS    = -g #-fsanitize=address -fno-omit-frame-pointer #-fno-inline
IFLAGS    = -Irapidjson/include/
LIBS      = -lm -lrt -lgsl -lblas
SOURCES   = $(wildcard *.cpp)
HEADERS   = $(wildcard *.hpp)

#########################################################################
# all:                                                                  #
#########################################################################
all: $(TARGETS)

#########################################################################
# asr_engine:                                                           #
#########################################################################
SIETILL_O	= SieTill.o\
						Alignment.o\
						Config.o\
						Corpus.o\
						FeedForwardLayer.o\
						IO.o\
						Lexicon.o\
						Mixtures.o\
						NetworkLayer.o\
						NeuralNetwork.o\
						NNTraining.o\
						OutputLayer.o\
						Recognizer.o\
						SignalAnalysis.o\
						TdpModel.o\
						Training.o\
						Util.o

sietill:	$(SIETILL_O)
		$(CXX) $(LDFLAGS) -o sietill $(SIETILL_O) $(DFLAGS) $(LIBS)

#########################################################################
# clean / clean_all:                                                    #
#########################################################################
.PHONY: clean clean_all

clean:
	rm -f core core.* *~ *.il *.o

clean_all: clean
	rm sietill

#########################################################################
# generate dependency file if necessary                                 #
#########################################################################
%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $(IFLAGS) $(DFLAGS) $< -o $@
