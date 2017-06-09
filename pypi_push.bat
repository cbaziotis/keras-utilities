RMDIR /S /Q build
RMDIR /S /Q keras_utilities.egg-info
RMDIR /S /Q dist

python setup.py sdist bdist_wheel

twine upload dist/*
for /r %%f in (dist\*) do echo twine register %%f