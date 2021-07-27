install:
	pip3 install -Ur requirements.txt
	pip3 install -q -U git+https://github.com/mljar/mljar-supervised.git@master

remove:
	pip3 uninstall -r requirements.txt

clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete