help: ## Print this message.
	@echo "\033[0;1mCommands\033[0m"
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[34;1m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: help

container.sif: container.def requirements.txt dask_config.yml ## The Singularity container. Requires sudo to run.
	singularity build $@ $<

shell: ## Start a shell in the container.
	singularity shell --cleanenv --nv container.sif
shell-bind: ## Start a shell with GUI and with ./results bound to /results.
	singularity shell --cleanenv --nv --bind ./results:/results container.sif
.PHONY: shell shell-gui

test: ## Run unit tests.
	pytest src/
.PHONY: test
