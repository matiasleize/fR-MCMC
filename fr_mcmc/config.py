'''
Here it is required to specify the config yml file that will be used.

TODO: see section "Ensure portability by using environment variables".
'''

from box import Box
import yaml
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git + '/configs/')

#Here you have to specify the the name of your .yml file

##LCDM
##yml_file = 'config_LCDM_PPS.yml'
##yml_file = 'config_LCDM_CC.yml'
##yml_file = 'config_LCDM_BAO_full.yml'
##yml_file = 'config_LCDM_PPS+CC.yml'
##yml_file = 'config_LCDM_PPS+BAO_full.yml'
##yml_file = 'config_LCDM_CC+BAO_full.yml'
##yml_file = 'config_LCDM_PPS+CC+BAO_full.yml'

##HS
#yml_file = 'config_HS_PPS.yml'
#yml_file = 'config_HS_CC.yml'
yml_file = 'config_HS_BAO_full.yml'
#yml_file = 'config_HS_PPS+CC.yml'
#yml_file = 'config_HS_PPS+BAO_full.yml'
#yml_file = 'config_HS_CC+BAO_full.yml'
#yml_file = 'config_HS_PPS+CC+BAO_full.yml'

##ST
#yml_file = 'config_ST_PPS.yml'
#yml_file = 'config_ST_CC.yml'
#yml_file = 'config_ST_BAO_full.yml'
#yml_file = 'config_ST_PPS+CC.yml'
#yml_file = 'config_ST_PPS+BAO_full.yml'
#yml_file = 'config_ST_CC+BAO_full.yml'
#yml_file = 'config_ST_PPS+CC+BAO_full.yml'

##EXP
#yml_file = 'config_EXP_PPS.yml'
#yml_file = 'config_EXP_CC.yml'
#yml_file = 'config_EXP_BAO_full.yml'
#yml_file = 'config_EXP_PPS+CC.yml'
#yml_file = 'config_EXP_PPS+BAO_full.yml'
#yml_file = 'config_EXP_CC+BAO_full.yml'
#yml_file = 'config_EXP_PPS+CC+BAO_full.yml'

with open(yml_file, "r") as ymlfile:
    full_cfg = yaml.safe_load(ymlfile)
    
cfg = Box({**full_cfg}, default_box=True, default_box_attr=None)
