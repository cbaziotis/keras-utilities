from setuptools import setup, find_packages

setup(name='keras-utilities',
      version='0.1.1',
      description='Utilities for Keras.',
      url='https://github.com/cbaziotis/keras-utilities',
      author='Christos Baziotis',
      author_email='christos.baziotis@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['docs', 'tests*']),
      install_requires=["keras==1.2.1", "matplotlib", "numpy", "seaborn", "scikit_learn"],
      include_package_data=True
      )
