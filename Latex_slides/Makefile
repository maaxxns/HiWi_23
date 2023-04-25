all: build/slides.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/slides.pdf: FORCE | build
	latexmk $(texoptions) slides.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc slides.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
