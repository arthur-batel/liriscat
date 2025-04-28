.PHONY: install clean check_conda check_env_file

install: check_conda check_env_file
	@if conda env list | grep -q 'liriscat-env'; then \
	    echo "Updating existing conda environment 'liriscat-env'..."; \
	    conda env update -f environment.yaml -n liriscat-env; \
	else \
	    echo "Creating new conda environment 'liriscat-env'..."; \
	    conda env create -f environment.yaml -n liriscat-env; \
	fi
	@echo "Installing package in editable mode..."
	pip install -e .

clean:
	@echo "Cleaning data and results directories..."
	rm -rf data/ results/

check_conda:
	@if command -v conda >/dev/null 2>&1; then \
		echo "Conda is installed."; \
	else \
		echo "ERROR: Conda is not installed. Please install Conda and run the Makefile again."; \
		exit 1; \
	fi

check_env_file:
	@if [ -f environment.yaml ]; then \
		echo "Found environment.yaml."; \
	else \
		echo "ERROR: environment.yaml file not found!"; \
		exit 1; \
	fi
