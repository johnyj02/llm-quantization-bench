# IEEE conference version (MLNLP 2026). Shares tables, figures and the bibliography with ../paper.
#
#   make          compile main.pdf
#   make check    compile, then refuse on placeholders, unresolved references or TODOs

TECTONIC ?= /opt/homebrew/bin/tectonic

.PHONY: all check

all: main.pdf

main.pdf: main.tex ../paper/custom.bib $(wildcard ../paper/tables/*.tex)
	$(TECTONIC) -X compile main.tex --outdir . --keep-intermediates

check: main.pdf
	@pdftotext main.pdf - > .main.txt
	@if grep -n -E "PENDING|Anonymous|\?\?|\[\?\]|TODO:" .main.txt; then \
	  echo "NOT SUBMITTABLE: placeholder, unresolved reference or TODO above"; rm -f .main.txt; exit 1; fi
	@rm -f .main.txt; echo "pages: $$(pdfinfo main.pdf | awk '/^Pages/ {print $$2}')  SUBMITTABLE"
