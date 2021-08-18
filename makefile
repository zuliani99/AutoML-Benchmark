install:
	pip3 install -Ur requirements.txt

remove:
	pip3 uninstall -r requirements.txt -y

clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete