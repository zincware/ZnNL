# ZnRND makefile for ZnRND

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
APIBUILD	  = sphinx-apidoc

define copy_notebooks
	mkdir source/examples/_notebooks/
	cp ../examples/* source/examples/_notebooks/
endef

define clean_generated_files
	rm -r source/examples/_notebooks
endef

# Help command is called first
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target. Implements the following:
# 		1.) Build the modules .rst files
#		2.) Copies example notebooks into the relevant directory
#		3.) Build the documentation
%: Makefile
	@$(APIBUILD) -o ./source/_modules ../
	$(call copy_notebooks)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	$(call clean_generated_files)
