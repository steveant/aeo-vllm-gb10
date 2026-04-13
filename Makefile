.PHONY: clean validate build install uninstall

clean:
	uv run scripts/make_clean.py

validate:
	uv run scripts/make_validate.py

build: validate
	uv run scripts/make_build.py

install: build  ## requires sudo
	@./dist/bootstrap-vllm --version
	cp dist/bootstrap-vllm /usr/local/bin/
	@echo "Installed: /usr/local/bin/bootstrap-vllm"

uninstall:  ## requires sudo
	rm -f /usr/local/bin/bootstrap-vllm
	@echo "Uninstalled: /usr/local/bin/bootstrap-vllm"
