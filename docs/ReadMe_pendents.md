# About frontend modifications.

## Some ideas
Instead of inputs on the .yml files with names True or False, use any word that starts with T or F.

# About backend modifications
Based on: https://medium.com/bcggamma/data-science-python-best-practices-fdb16fdedf82

### Directories distribution
'''
tesis_licenciatura/fr_mcmc --> supermodel/supermodel
supermodel/supermodel/tests/: this have conatin all that appear in the section "Unit test a lot" 
'''

### Other stuff
Rename solve_sys as solve_sist (grep if it is necessary to change this in other location).

Add to environment.yml: PythonBox, seaborn, getdist (maybe something else)

All data should be in Pandas df format (maybe).

Make classes for sampler and data manager (see utils/datos.py).