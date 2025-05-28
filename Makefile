.PHONY: uv-clean uv-sync

uv-clean:
	rm -f uv.lock
	rm -rf .venv
	rm -rf ~/.cache/uv/
	rm -rf build/ dist/ *.egg-info

uv-sync: uv-clean
	uv sync --no-cache --no-build-isolation --extra build
