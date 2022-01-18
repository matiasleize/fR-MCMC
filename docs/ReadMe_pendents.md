# ReadMe sobre las modificaciones al backend
Basado en la entrada: https://medium.com/bcggamma/data-science-python-best-practices-fdb16fdedf82

## Ideas del backend
Aca si vamos a trabajar en un futuro con clases y esas cosas. En esta parte solo nos enfocamos en la experiencia del usuario. Mismo podemos hacer varias versiones del code.py si precisamos alguna modificacion temporal, ademas de tener varios config.yml. Quiza en el futuro moverlos al home ddel repositorio. Ver como se estructura en el tutorial que recomienda cierta jerarquia. 

### Distribucion de directorios
tesis_licenciatura/Software --> supermodel/supermodel
tesis_licenciatura/Software/Estadística/data --> supermodel/supermodel/source 
tesis_licenciatura/Software/Estadística/MCMC --> supermodel/supermodel/model 
tesis_licenciatura/Software/Funcionales --> supermodel/supermodel/utils
tesis_licenciatura/Software/Sandbox --> supermodel/supermodel/tests/checks
tesis_licenciatura/Software/Estadística/MCMC/config...yml --> supermodel/configs/config...yml

supermodel/supermodel/tests/checks: aca poner todo lo que esta en la seccion "Unit test a lot" 


(El resto de los directorios que aparecen en la página no creo que sean necesarios)
