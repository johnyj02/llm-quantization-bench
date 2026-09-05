# Paper build + gate G6 check.
#
# tectonic is a single self-contained binary (~50MB cache) that downloads TeX
# packages on demand -- it is why we do NOT need MacTeX (~7GB, which does not
# fit) and are NOT blocked on Overleaf. See notes/paper-toolchain.md.
#
#   make          compile main.pdf
#   make g6        check gate G6: compiles, <=8 pages, \todo{} only in Results
#   make clean

TECTONIC ?= /opt/homebrew/bin/tectonic
MAIN     := main
PAGE_LIMIT := 8

.PHONY: all g6 pages clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex custom.bib acl.sty acl_natbib.bst
	@# --keep-intermediates because G6 reads the BODY page count out of main.aux.
	@# Without it tectonic discards the aux, and a STALE aux from an older build sat
	@# here for weeks looking authoritative: 28 resolved labels, none of them current.
	$(TECTONIC) -X compile $(MAIN).tex --outdir . --keep-intermediates

pages: $(MAIN).pdf
	@python3 -c "import re,zlib;d=open('$(MAIN).pdf','rb').read();\
t=0;\
[t:=t+len(re.findall(rb'/Type\s*/Page[^sC]',zlib.decompressobj().decompress(d[d.find(b chr(10),m.end())+1:d.find(b'endstream',m.end())]))) for m in re.finditer(rb'stream',d)];\
print(t)" 2>/dev/null || python3 ../scripts/pdf_pages.py $(MAIN).pdf

# Gate G6 (Sep 20): compiles, <= 8 pages, every section written,
# \todo{} permitted ONLY inside Results.
g6: $(MAIN).pdf
	@echo "== G6: compiles =="
	@test -f $(MAIN).pdf && echo "  PASS"
	@echo "== G6: BODY page count <= $(PAGE_LIMIT) =="
	@# 🚨 BODY pages, not total. ACL counts Limitations, Ethics, Acknowledgments,
	@# references and appendices as ZERO page budget, so comparing the total against 8
	@# over-counts and would force cuts to body content that never competed for the space.
	@b=$$(python3 ../scripts/body_pages.py $(MAIN).aux); \
	 t=$$(python3 ../scripts/pdf_pages.py $(MAIN).pdf); \
	 echo "  body pages: $$b   (total PDF incl. unlimited sections: $$t)"; \
	 if [ "$$b" -le $(PAGE_LIMIT) ]; then echo "  PASS"; else echo "  FAIL"; exit 1; fi
	@echo "== G6: required sections present AND non-empty =="
	@# 🚨 RESTORED 2026-09-05. This was `grep -q 'section*{Limitations}'`, which tests only
	@# that the HEADING exists, and commit 4a44efe removed the replacement along with an
	@# unrelated scope-audit revert. While it was absent, `make g6` reported PASS on this
	@# criterion with five required sections carrying zero prose, Limitations among them.
	@# An empty Limitations section is an automatic desk reject exactly as a missing one is.
	@python3 ../scripts/check_sections.py $(MAIN).tex
	@printf '== G6: todo markers only inside Results ==\n'
	@python3 ../scripts/check_todo_scope.py $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).out $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).synctex.gz $(MAIN).fdb_latexmk $(MAIN).fls
