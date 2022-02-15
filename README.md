# Tesis de Licenciatura
Tesis de la Licenciatura en Ciencias Físicas. Facultad de Ciencias Exactas y Naturales de la Universidad de Buenos Aires.

## Create a virtual environment
In order to create a virtual environment with the libraries that are needed to run this module, follow the next steps:
(1) Clone the repository: $ git clone 
(2) Run: $ conda env create 
(3) Run: $ source activate tesis_licenciatura

## Create an output directory:
Output files can be particarly heavy stuff. For instance, the markov chains are saved in h5 format of several MegaBites. To avoid the unnecessary use of memory on the main repository, output files are stored on an independent directory on the computer's user. For default, this file must be created on the same directory that the Git's repository was cloned:

root_directory/                    Root directory
├── tesis_licenciatura/            Root project directory
├── output/                        Output directory

Having said that, the user can change the location of the ouput directory on the configuration file.

## Configuration file:
The file config.py (which is located in the directory tesis_icenciatura/configs) shows all the configuration parameters. 
