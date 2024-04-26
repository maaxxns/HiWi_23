thz: *.csv
test: test.pdf
FFT: *.txt.pdf

*.txt.pdf: src/FFT_and_stuff.py | build
	python src/FFT_and_stuff.py

*.csv: src/THz.py | build
	python src/THz.py

test.pdf: src/THz_testing.py | build 
	python src/THz_testing.py

build:
	mkdir -p build
	mkdir -p build/testing
	mkdir -p build/FFTs
	mkdir -p build/testing/Transfertest_Thz
	mkdir -p build/testing/transferfunction_test
	mkdir -p build/results
clean:
	rm -r build