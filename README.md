# Analisis estadísitco de modelos $f(R)$
Autor: Matías Leizerovich. Facultad de Ciencias Exactas y Naturales, Universidad de Buenos Aires.

Para descargar el repositorio, ver https://github.com/matiasleize/fR-MCMC

Esta versión del código fue desarrollada para realizar los analisis que se encuentran en el paper 'Testing f(R) gravity models with quasar x-ray and UV fluxes', by M. Leizerovich, L. Kraiselburd, S. Landau and C. Scóccola, publicado en Phys. Rev. D 105, 103526 (2022). Ver https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.103526. Puedes usar fR-MCMC de manera libre, siempre que cites el paper mencionado anteriormente.

## Observación importante:
Para ejecutar correctamente un archivo de este repositorio, se debe editar el archivo 'pc_path.py' que se encuentra en el mismo directorio. En particular, se deben agregar el 'hostname' de la computadora local, detallando cual es la carpeta en donde se clona el repositorio ('path_git') y donde se desea que se almacenen los resultados ('path_datos_global'), demasiado pesados para almacenarlos en el repositorio.

## Crear un ambiente virtual
Para crear un enteorno virtual con las librerías necesarias para correr los códigos de este repositorio, seguir los siguientes pasos:
(1) Clonar este repositorio
(2) Correr el comando: $ conda env create 
(3) Correr el comando: $ source activate fR-MCMC
