# Assignment 1 - basic matrix multiplication in C, Java and Python
# Every step can also be run by hand; see README.md.

PYTHON ?= python3
CC      = gcc
CFLAGS  = -O2 -std=c11 -ffp-contract=off

.PHONY: all build inputs env validate bench diag analyze report clean

all: build inputs env validate bench diag analyze

build: build/matmul_c build/java/MatMul.class

build/matmul_c: src/c/matmul.c
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ $< -lm

build/java/MatMul.class: src/java/MatMul.java
	mkdir -p build/java
	javac -d build/java $<

inputs:
	$(PYTHON) scripts/generate_inputs.py

env:
	$(PYTHON) scripts/env_info.py

validate: build
	$(PYTHON) scripts/validate.py

bench: build
	$(PYTHON) scripts/run_benchmarks.py

diag:
	bash scripts/run_diagnostic.sh

analyze:
	$(PYTHON) scripts/analyze.py

report:
	cd report && pandoc report.md -o ../Assignment01_PerezSuarez_Report.pdf --pdf-engine=xelatex

clean:
	rm -rf build data/*.bin
