.PHONY: all clean

all: main.pyc

main.pyc: main.py
	python3 -m compileall -b main.py

clean:
	rm -f __pycache__/main.cpython-*.pyc
	rm -rf __pycache__
