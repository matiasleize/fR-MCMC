# $f(R)$ models statistical analysis
Author: Matías Leizerovich. Faculty of Exact and Natural Sciences, Buenos Aires University.

For download, see https://github.com/matiasleize/fR-MCMC

This version of the code was developed to make the analysis of the paper 'Testing f(R) gravity models with quasar x-ray and UV fluxes', by M. Leizerovich, L. Kraiselburd, S. Landau and C. Scóccola, published in Phys. Rev. D 105, 103526 (2022). See https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.103526. You can use fR-MCMC freely, provided that in your publications you cite the paper mentioned.

## Create a virtual environment
In order to create a virtual environment with the libraries that are needed to run this module, follow the next steps:
* Clone the repository: ```git clone https://github.com/matiasleize/fR-MCMC``` 
* Enter the directory: ```cd fR-MCMC```
* Create the virtual environment: ```conda env create``` 
* Activate the virtual environment: ```source activate fR-MCMC```

## Create an output directory:
Output files can be particularly heavy stuff. For instance, the markov chains are saved in h5 format of several MegaBites. To avoid the unnecessary use of memory in the main repository, output files are stored in an independent directory in the computer's user. For default, this file must be created in the same directory that the Git's repository was cloned:

```
root_directory/              Root directory
├── fR-MCMC/                 Root project directory
├── fR-output/               Output directory
```

Having said that, the user can change the location of the ouput directory on the configuration file.

## Configuration file:
The files (.yml) located in the directory fR-MCMC/configs shows all the configuration parameters. 

## Run the code:
To run the code for a particular configuration file, edit config.py (which is located in the directory fR-MCMC/fr_mcmc) and then run the following command while you are on the root project directory:  

```
python3 -m fr_mcmc --task mcmc
```

If it is desired to run only the analyses part of the code over an existing Markov Chain result, run:

```
python3 -m fr_mcmc --task analysis --outputfile 'filename'
```

where 'filename' is the name of the directory where the runs are stored (as an example: 'filename' =  'sample_HS_SN_CC_4params').
