.PHONY: install clean check_conda check_env_file

install: check_conda check_env_file
	@if conda env list | grep -q 'micat-env'; then \
	    echo "Updating existing conda environment 'micat-env'..."; \
	    conda env update -f environment.yml -n micat-env; \
	else \
	    echo "Creating new conda environment 'micat-env'..."; \
	    conda env create -f environment.yml -n micat-env; \
	fi
	@echo "Installing package in editable mode..."
	pip install -e .
	pip install -e ../IMPACT

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
	@if [ -f environment.yml ]; then \
		echo "Found environment.yml."; \
	else \
		echo "ERROR: environment.yml file not found!"; \
		exit 1; \
	fi
