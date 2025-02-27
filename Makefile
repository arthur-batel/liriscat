.PHONY: install, clean, check_conda

install: check_conda

	conda init
	conda env create -f environment.yaml -n liriscat-env \
  || conda env update -f environment.yaml -n liriscat-env;

clean:
	rm -rf data/
	rm -rf results/

check_conda:
	@if command -v conda >/dev/null 2>&1; then \
		echo "conda is installed"; \
	else \
		echo "conda needs to be installed\nrun the makefile again after the installation"; \
		exit 1; \
	fi