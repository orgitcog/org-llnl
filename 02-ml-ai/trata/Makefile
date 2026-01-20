SHELL := /bin/bash

USER_WORKSPACE := $(if $(USER_WORKSPACE), $(USER_WORKSPACE),/usr/workspace/$(USER))
WORKSPACE = $(USER_WORKSPACE)/gitlab/weave/trata
TRATA_ENV := $(if $(TRATA_ENV),$(TRATA_ENV),trata_env)
 
CI_UTILS = /usr/workspace/weave/ci_utils

PYTHON_CMD = /usr/tce/bin/python3

PIP_OPTIONS = --trusted-host wci-repo.llnl.gov --index-url https://wci-repo.llnl.gov/repository/pypi-group/simple --use-pep517

DOCS_PKGS = sphinx nbsphinx nbconvert furo

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

define run_trata_tests
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
	@echo "Create venv for running trata...$(WORKSPACE)";
	mkdir -p $(WORKSPACE);
	cd $(WORKSPACE);
	if [ -d $(TRATA_ENV) ]; then \
	  rm -rf $(TRATA_ENV); \
	fi;
	$(call create_env,$(WORKSPACE)/$(TRATA_ENV))


.PHONY: run_tests
run_tests:
	@echo "Run tests...";
	$(call run_trata_tests,$(WORKSPACE)/$(TRATA_ENV))


.PHONY: build_docs
build_docs:
	@echo "Build docs...";
	source $(WORKSPACE)/$(TRATA_ENV)/bin/activate && \
	pip install $(PIP_OPTIONS) $(DOCS_PKGS) && \
	cd docs && make html

# .PHONY: release
# release:
# 	@echo "...create a release....TAG: $(CI_COMMIT_TAG), PKG_REGISTRY_URL: $(PKG_REGISTRY_URL)"; \
# 	$(eval TAG=$(shell  echo $(CI_COMMIT_TAG) | sed -e "s/^trata-//"))
# 	env; \
# 	$(CI_UTILS)/bin/release-cli create --name "Trata $(CI_COMMIT_TAG)" --tag-name $(CI_COMMIT_TAG); \
# 	tar -cvf $(TAG).tar trata; \
# 	ls; \
# 	gzip $(TAG).tar; \
# 	curl --header "JOB-TOKEN: $(CI_JOB_TOKEN)" --upload-file $(TAG).tar.gz $(PKG_REGISTRY_URL)/$(CI_COMMIT_TAG)/$(TAG).tar.gz


# .PHONY: deploy
# .ONESHELL:
# deploy:
# 	@echo "...deploy...only run from CI... "; \
# 	$(eval TAG=$(shell  echo $(CI_COMMIT_TAG) | sed -e "s/^trata-//"))
# 	wget --header="JOB-TOKEN:$(CI_JOB_TOKEN)" $(PKG_REGISTRY_URL)/$(CI_COMMIT_TAG)/$(TAG).tar.gz -O $(TAG).tar.gz
# 	give weaveci $(TAG).tar.gz
# 	xsu weaveci -c "sg us_cit" <<AS_WEAVECI_USER
# 		cd $(DEPLOY_PATH)
# 		take muryanto -f
# 		chmod 750 $(TAG).tar.gz
# 		gunzip $(TAG).tar.gz
# 		tar -xvf $(TAG).tar
# 		rm $(TAG).tar
# 		mv trata $(TAG)
# 		chmod -R 750 $(TAG)
# 		rm -f current
# 		ln -s $(TAG) current
# 		sed -i 's|python|$(PYTHON_CMD)|' $(TAG)/trata
# 	AS_WEAVECI_USER


# .PHONY: deploy_to_develop
# .ONESHELL:
# deploy_to_develop:
# 	$(eval VERSION=`cat $(CI_PROJECT_DIR)/trata/scripts/version.txt`)
# 	echo "...deploy_to_develop...VERSION: $(VERSION)"
# 	cd trata && if [ -d __pycache__ ]; then rm -rf __pycache__; fi
# 	if [ -f $(VERSION).tar.gz ]; then rm -f $(VERSION).tar.gz; fi 
# 	tar -cvf $(VERSION).tar * && gzip $(VERSION).tar
# 	give --force weaveci $(VERSION).tar.gz
# 	xsu weaveci -c "sg us_cit" <<AS_WEAVECI_USER
# 		umask 027
# 		cd $(DEPLOY_PATH)
# 		if [ ! -d $(DEPLOY_PATH)/develop ]; then mkdir -p $(DEPLOY_PATH)/develop; fi
# 		cd $(DEPLOY_PATH)/develop
# 		take muryanto -f
# 		gunzip $(VERSION).tar.gz
# 		tar -xvf $(VERSION).tar && rm $(VERSION).tar
# 		cd .. && chmod -R 750 develop
# 		sed -i 's|python|$(PYTHON_CMD)|' develop/trata
# 	AS_WEAVECI_USER
