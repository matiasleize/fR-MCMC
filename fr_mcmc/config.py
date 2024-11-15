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
yml_file = 'config_LCDM_4p_PPS+CC+BAO.yml'
#yml_file = 'config_HS_5p_PPS+CC+BAO.yml'

with open(yml_file, "r") as ymlfile:
    full_cfg = yaml.safe_load(ymlfile)
    
cfg = Box({**full_cfg}, default_box=True, default_box_attr=None)


## OLD CODE
#yml_file = 'config.yml'
#yml_file = 'config_test_CC_IOI_LCDM_5.yml' 
#yml_file = 'config_test_CC_IOI_HS_5.yml' 
#yml_file = 'config_test_CC_IOI_LCDM_8.yml' 
#yml_file = 'config_test_CC_IOI_HS_8.yml' 
#yml_file = 'config_CC.yml' 
#yml_file = 'config_SN+H0.yml' 
#yml_file = 'config_PPS.yml'
#yml_file = 'config_PPS+CC.yml'
#yml_file = 'config_PPS+CC+BAO.yml'

#yml_file = 'config_PPS_ASTRO.yml'
#yml_file = 'config_PP_CM.yml'
