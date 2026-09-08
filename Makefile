.DEFAULT_GOAL := help
PYTHON ?= python3
FIGURES := loss_curve_comparison accuracy_curve_comparison centralized_vs_federated_bar \
	centralized_confusion_matrix federated_confusion_matrix centralized_roc_curve federated_roc_curve \
	centralized_example_predictions federated_example_predictions \
	centralized_gradcam_examples federated_gradcam_examples

.PHONY: help install install-dev lint test train train-fed evaluate figures clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with runtime dependencies
	$(PYTHON) -m pip install -e .

install-dev:  ## Install the package with development tooling
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Run ruff
	ruff check .

test:  ## Run the test suite
	pytest

train:  ## Train the centralized baseline
	xray-fl train-centralized

train-fed:  ## Train the FedAvg federated model
	xray-fl train-federated

evaluate:  ## Evaluate saved checkpoints and regenerate figures
	xray-fl evaluate

figures:  ## Refresh the README figures in docs/images from outputs/plots
	cp $(addprefix outputs/plots/,$(addsuffix .png,$(FIGURES))) docs/images/

clean:  ## Remove generated artifacts and caches
	rm -rf outputs .pytest_cache .ruff_cache
