# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import bender

setup(
   name='Bender',
   version=bender.__version__,
   packages=find_packages(),
   author_email='benjamin@rythm.co',
   description='Bender Python Client',
   long_description=open('README.md').read(),
   install_requires=['requests', ],
   include_package_data=True,
   url='https://github.com/Dreem-Devices/Bender-Client',
)
