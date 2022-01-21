# ReadMe sobre las modificaciones al backend
Basado en la entrada: https://medium.com/bcggamma/data-science-python-best-practices-fdb16fdedf82

## Ideas del backend
Aca si vamos a trabajar en un futuro con clases y esas cosas. En esta parte solo nos enfocamos en la experiencia del usuario. Mismo podemos hacer varias versiones del model.py si precisamos alguna modificacion temporal, ademas de tener varios config.yml. Quiza en el futuro moverlos a la carpeta configs del repositorio. 

### Distribucion de directorios
tesis_licenciatura/Software --> supermodel/supermodel
tesis_licenciatura/Software/Estadística/data --> supermodel/supermodel/source (done)
tesis_licenciatura/Software/Estadística/MCMC --> supermodel/supermodel/model (done)
tesis_licenciatura/Software/Funcionales --> supermodel/supermodel/utils
tesis_licenciatura/Software/Sandbox --> supermodel/supermodel/tests/checks (done)
tesis_licenciatura/Software/Estadística/MCMC/config...yml --> supermodel/configs/config...yml (semi done)

supermodel/supermodel/tests/: aca poner todo lo que esta en la seccion "Unit test a lot" 
(El resto de los directorios que aparecen en la página no creo que sean necesarios)

### Otras cosas
Quiza mover a test tambien las funciones viejas o que no son de uso diario (por ejemplo los otros integradores). Luego, Acordarse de cambiar de nombre int_1 por int, tanto en funciones_alternativos como en funciones_parametros_derivados )y searchear si en otro lugar).

Volver a poner el archivo de texto que cambia directamente el environment, quiza eso arregla el error del ambiente virtual.

Agregar al environment.yml PythonBox (y quiza alguna cosa mas)