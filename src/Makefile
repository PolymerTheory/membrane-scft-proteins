# Makefile for compiling CUDA code

CC ?= nvcc
NVCC ?= nvcc
CFLAGS=-O3
LIBS=-lmpi -lcufft
LDFLAGS=-L$(EBROOTCUDA)/lib64
TARGET=scft_gpu
SOURCE=scft.cu
PREVIEW_TARGET=protein_preview
PREVIEW_SOURCE=protein_preview.cpp
PROTEIN_HEADER ?= protein_arcs.h
SCFT_STRN ?= 48
USE_HDF5 ?= 1
HDF5_DIR ?= $(or $(EBROOTHDF5),$(HDF5_ROOT))

CFLAGS += -DPROTEIN_HEADER=\"$(PROTEIN_HEADER)\"
CFLAGS += -DSCFT_STRN=$(SCFT_STRN)

ifeq ($(USE_HDF5),1)
	CFLAGS += -DUSE_HDF5
	ifneq ($(strip $(HDF5_DIR)),)
		CFLAGS += -I$(HDF5_DIR)/include
		LDFLAGS += -L$(HDF5_DIR)/lib
	endif
	LIBS += -lhdf5 -lz
endif

all: $(TARGET)

$(TARGET): $(SOURCE) $(wildcard *.h) FORCE
	$(NVCC) $(CFLAGS) $(LDFLAGS) $(LIBS) -o $(TARGET) $(SOURCE)

$(PREVIEW_TARGET): $(PREVIEW_SOURCE) $(wildcard *.h) FORCE
	c++ -O2 -DPROTEIN_HEADER=\"$(PROTEIN_HEADER)\" -DSCFT_STRN=$(SCFT_STRN) -o $(PREVIEW_TARGET) $(PREVIEW_SOURCE)

clean:
	rm -f $(TARGET) $(PREVIEW_TARGET)

.PHONY: all clean FORCE
FORCE:
