# Revised combination (correlation tempering + (M, C) space). See README.md.
#
#   make numbers   print headline M, R, C, P_i, ... from saved results (seconds)
#   make results   run the (M, C) combination if missing or older than core.py
#                  (~2 min; `make -B results` forces a rerun)
#   make figures   remake the figures in revised_method/figures/
#   make all       results + figures + numbers
#
# The grid is the manuscript notebook's, M 0.95-2.2 Msun (core.Setup).
# Use another interpreter with e.g. `make all PYTHON=python3.11`.

PYTHON ?= python3
DIR := revised_method
RES := $(DIR)/results

.PHONY: all results figures numbers

all: results figures numbers

$(RES)/combined_MC.npz: $(DIR)/core.py
	$(PYTHON) $(DIR)/combine.py

results: $(RES)/combined_MC.npz

figures: results
	$(PYTHON) $(DIR)/plots.py

numbers:
	@$(PYTHON) $(DIR)/headline.py
