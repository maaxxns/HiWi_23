num: THzPhase.pdf

FFT: *.txt.pdf

*.txt.pdf: src/FFT_and_stuff.py | build
	python src/FFT_and_stuff.py

THzPhase.pdf: src/THz.py | build 
	python src/THz.py

build:
	mkdir -p build
	mkdir -p build/testing
	mkdir -p build/FFTs

clean:
	rm -r build