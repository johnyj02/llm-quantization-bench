# IEEE conference version (MLNLP 2026). Shares tables, figures and the bibliography with ../paper.
#
#   make          compile main.pdf (full) and main6.pdf (six-page)
#   make check    compile both, then refuse on placeholders, unresolved references or TODOs

TECTONIC ?= /opt/homebrew/bin/tectonic

.PHONY: all check

all: main.pdf main6.pdf

%.pdf: %.tex ../paper/custom.bib $(wildcard ../paper/tables/*.tex)
	$(TECTONIC) -X compile $< --outdir . --keep-intermediates

check: main.pdf main6.pdf
	@for f in main main6; do \
	  pdftotext $$f.pdf - > .$$f.txt; \
	  if grep -n -E "PENDING|Anonymous|\?\?|\[\?\]|TODO:" .$$f.txt; then \
	    echo "$$f.pdf NOT SUBMITTABLE: placeholder, unresolved reference or TODO above"; rm -f .$$f.txt; exit 1; fi; \
	  rm -f .$$f.txt; echo "$$f.pdf  pages: $$(pdfinfo $$f.pdf | awk '/^Pages/ {print $$2}')  SUBMITTABLE"; \
	done
