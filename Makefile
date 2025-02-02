.PHONY: prepare clean test format wikipage extract_relation

VERSION?=1.0.0
GIT_COMMIT=$$(git rev-parse --short HEAD)
BUILD_TIME=$$(date +%FT%T%z)
LDFLAGS=-ldflags "-X main.Version=${VERSION} -X main.GitCommit=${GIT_COMMIT} -X main.BuildTime=${BUILD_TIME}"

PYTHON_DIR=python
UV = uv
UV_RUN = $(UV) run

prepare:
	$(UV) sync --all-extras --dev
clean:
	$(UV) cache clean
test:
	$(UV_RUN) pytest
format:
	$(UV_RUN) ruff check --fix
	$(UV_RUN) ruff format
