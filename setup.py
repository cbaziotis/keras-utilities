from setuptools import setup, find_packages

setup(name='keras-utilities',
      version='0.4.0',
      description='Utilities for Keras.',
      url='https://github.com/cbaziotis/keras-utilities',
      author='Christos Baziotis',
      author_email='christos.baziotis@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['docs', 'tests*']),
      include_package_data=True
      )
