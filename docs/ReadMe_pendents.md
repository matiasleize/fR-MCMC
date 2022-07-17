# About frontend modifications.

## Some ideas
Instead of inputs on the .yml files with names True or False, use any word that starts with T or F.

# About backend modifications
Based on: https://medium.com/bcggamma/data-science-python-best-practices-fdb16fdedf82

## Some ideas
Aca si vamos a trabajar en un futuro con clases y esas cosas. En esta parte solo nos enfocamos en la experiencia del usuario. Mismo podemos hacer varias versiones del model.py si precisamos alguna modificacion temporal, ademas de tener varios config.yml.

### Directories distribution
'''
tesis_licenciatura/Software --> supermodel/supermodel
supermodel/supermodel/tests/: this have conatin all that appear in the section "Unit test a lot" 
'''

### Other stuff
Move the old functions or the ones that are not periodically used to the test folder (for instance, the other integrators). Important: rename int_1 as int, not only in alternativos but also in parametros_derivados (grep if it is necessary to change this in other location).

Add to environment.yml: PythonBox, seaborn, getdist (maybe something else)

All data should be in Pandas df format (maybe).

Make classes for sampler and data manager (see utils/datos.py).
