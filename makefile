install:
	pip install -Ur requirements.txt

remove:
	pip uninstall -r requirements.txt

clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete