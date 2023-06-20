all: build/slides.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/slides.pdf: FORCE | build
	latexmk $(texoptions) slides_presentation.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc slides_presentation.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
