# Makefile for earplug~

lib.name = earplug~

class.sources = earplug~.c

# set this to disable embedding the default impulse response data set
#cflags = -DEARPLUG_DATA_NO_EMBED

datafiles = earplug~-help.pd earplug~-meta.pd earplug_data.h earplug_data_compensated.txt \
            parse-to-h.py README.txt LICENSE.txt

PDLIBBUILDER_DIR=.
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

all: earplug_data_compensated.txt

PYTHON ?= python3
PLOT_DIR ?= curves
PLOT_PREFIX ?= $(PLOT_DIR)/hrir_norm

earplug_data_compensated.txt: normalize_hrir.py earplug_data.txt
	mkdir -p $(PLOT_DIR)
	$(PYTHON) normalize_hrir.py --input earplug_data.txt --output $@ --plot-prefix $(PLOT_PREFIX)

.PHONY: compensated-data
compensated-data: earplug_data_compensated.txt
