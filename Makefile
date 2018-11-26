all: build deploy clean

build:
	python setup.py sdist bdist_wheel
deploy:
	twine upload dist/* --verbose
clean:
	rm -rf dist build bender_client.egg-info