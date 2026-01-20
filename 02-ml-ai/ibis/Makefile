SHELL := /bin/bash

USER_WORKSPACE := $(if $(USER_WORKSPACE),$(USER_WORKSPACE),/usr/workspace/$(USER))
WORKSPACE = $(USER_WORKSPACE)/gitlab/weave/ibis
IBIS_ENV := $(if $(IBIS_ENV),$(IBIS_ENV),ibis_env)

PIP_OPTIONS = --trusted-host wci-repo.llnl.gov --index-url https://wci-repo.llnl.gov/repository/pypi-group/simple --use-pep517

DOCS_PKGS = sphinx nbsphinx nbconvert sphinx-rtd-theme

# SOURCE_ZONE is set in CI variable
# Set SOURCE_ZONE to 'CZ', 'RZ' or 'SCF' on command line for manual testing
ifeq ($(SOURCE_ZONE),SCF)
    GITLAB_URL = $(SCF_GITLAB)
else ifeq ($(SOURCE_ZONE),RZ)
    GITLAB_URL = $(RZ_GITLAB)
else
    GITLAB_URL = $(CZ_GITLAB)
endif

BUILDS_DIR := $(if $(CI_BUILDS_DIR),$(CI_BUILDS_DIR)/gitlab/weave/ibis,$(shell pwd))

define create_env
	# call from the directory where env will be created
	# arg1: name of env
	if [ -d $1 ]; then rm -Rf $1; fi
	/usr/apps/weave/tools/create_venv.sh -p cpu -e $1 -v latest-develop
	source $1/bin/activate && \
	pip install . && \
	which pytest && \
	pip list
endef


define install_ibis
	cd $(BUILDS_DIR)
	source $1/bin/activate && \
	pip install $(PIP_OPTIONS) .  && \
	pip list && which pip
endef


define run_ibis_tests
	# call from the top repository directory
	# arg1: full path to venv
	source $1/bin/activate && \
	which pytest && \
	if [ $(TESTS) ]; then \
		pytest --capture=tee-sys -v $(TESTS); \
	else \
		pytest --capture=tee-sys -v tests/; \
	fi
endef


.PHONY: create_env
create_env:
	@echo "Create venv for running ibis...$(WORKSPACE)";
	mkdir -p $(WORKSPACE);
	cd $(WORKSPACE)
	if [ -d $(IBIS_ENV) ]; then \
		rm -rf $(IBIS_ENV); \
	fi
	$(call create_env,$(WORKSPACE)/$(IBIS_ENV))
	$(call install_ibis,$(WORKSPACE)/$(IBIS_ENV))

.PHONY: run_tests
run_tests:
	@echo "Run tests...";
	$(call run_ibis_tests,$(WORKSPACE)/$(IBIS_ENV))


.PHONY: build_docs
build_docs:
	@echo "Build docs...";
	source $(WORKSPACE)/$(IBIS_ENV)/bin/activate && \
	pip install $(PIP_OPTIONS) $(DOCS_PKGS) && \
	cd docs && make html


