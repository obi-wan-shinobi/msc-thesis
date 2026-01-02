SHELL := /bin/bash

PROJECT := thesis
SRC_DIR := src
BUILD_DIR := build

export TEXINPUTS := $(SRC_DIR)//:

TEX_MAIN := $(SRC_DIR)/$(PROJECT).tex

LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -bibtex -interaction=nonstopmode -halt-on-error \
                 -outdir=$(BUILD_DIR)

TEX_SOURCES := $(shell find $(SRC_DIR) -name '*.tex')
BIB_SOURCES := $(shell find bibliography -name '*.bib')
STYLE_SOURCES := $(shell find $(SRC_DIR) -name '*.sty')
IMG_SOURCES := $(shell find img -type f)

.PHONY: all pdf clean watch

all: pdf

pdf: $(BUILD_DIR)/$(PROJECT).pdf

$(BUILD_DIR)/$(PROJECT).pdf: $(TEX_MAIN) $(TEX_SOURCES) $(BIB_SOURCES) $(STYLE_SOURCES) $(IMG_SOURCES) Makefile
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX_MAIN)

watch:
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(TEX_MAIN)

clean:
	$(LATEXMK) -C -outdir=$(BUILD_DIR)
	rm -rf $(BUILD_DIR)
