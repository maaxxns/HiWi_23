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
	mkdir -p build/testing/Transfertest_Thz
	mkdir -p build/testing/transferfunction_test
clean:
	rm -r build