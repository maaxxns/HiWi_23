all: THzPhase.pdf

THzPhase.pdf: THz.py | build 
	python THz.py

build:
	mkdir -p build
	mkdir -p build/testing

clean:
	rm -r build