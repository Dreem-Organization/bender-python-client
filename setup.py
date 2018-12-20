# -*- coding: utf-8 -*-
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bender-client',
    version="0.3.2.0",
    packages=["benderclient"],
    author="Dreem",
    author_email='valentin@dreem.com',
    description='Bender Python Client',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bender.dreem.com",
    install_requires=[
        'requests==2.20.1',
        'python-jose==3.0.1',
    ],
)
