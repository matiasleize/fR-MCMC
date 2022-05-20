# f(R) models statistical analysis
Faculty of Exact and Natural Sciences, Buenos Aires University.

## Create a virtual environment
In order to create a virtual environment with the libraries that are needed to run this module, follow the next steps:
(1) Clone the repository: $ git clone 
(2) Run: ```conda env create``` 
(3) Run: ```source activate fR-MCMC```

## Create an output directory:
Output files can be particarly heavy stuff. For instance, the markov chains are saved in h5 format of several MegaBites. To avoid the unnecessary use of memory on the main repository, output files are stored on an independent directory on the computer's user. For default, this file must be created on the same directory that the Git's repository was cloned:

```
root_directory/                    Root directory
├── fR-MCMC/            Root project directory
├── results/                        Results/Output directory
```

Having said that, the user can change the location of the ouput directory on the configuration file.

## Configuration file:
The files (.yml) located on the directory fR-MCMC/configs shows all the configuration parameters. 

## Run the code:
To run the code for a particular configuration file, edit config.py (which is located on the directory fR-MCMC/Software) and then run the following command while you are on the root project directory:  

```
python3 -m Software --task mcmc
```
