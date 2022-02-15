# ReadMe sobre las modificaciones al backend
Basado en la entrada: https://medium.com/bcggamma/data-science-python-best-practices-fdb16fdedf82

## Ideas del backend
Aca si vamos a trabajar en un futuro con clases y esas cosas. En esta parte solo nos enfocamos en la experiencia del usuario. Mismo podemos hacer varias versiones del model.py si precisamos alguna modificacion temporal, ademas de tener varios config.yml.

### Distribucion de directorios
tesis_licenciatura/Software --> supermodel/supermodel

supermodel/supermodel/tests/: aca poner todo lo que esta en la seccion "Unit test a lot" 
(El resto de los directorios que aparecen en la página no creo que sean necesarios)

### Otras cosas
Quiza mover a test tambien las funciones viejas o que no son de uso diario (por ejemplo los otros integradores). Luego, Acordarse de cambiar de nombre int_1 por int, tanto en alternativos como en parametros_derivados )y searchear si en otro lugar).

Volver a poner el archivo de texto que cambia directamente el environment, quiza eso arregla el error del ambiente virtual.

Agregar al environment.yml PythonBox, seaborn, getdist (y quiza alguna cosa mas)

Hacer una carpeta con el posprocesado?  Seria raro xq en el main la llamarias solo si hay LCDM. Quiza en el modulo posprocesado poner un if que si es LCDM hacer "pass".

Quiza pasar todos los datos a formato df de pandas.

Preguntar que es ancillary_g10.txt
