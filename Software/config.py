'''
Esto actualizarlo con nuestra info. Ver sección "Ensure portability by using environment variables".
'''

from box import Box
import yaml
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git + '/configs/')

yml_file = 'config.yml'
yml_file = 'config_HS.yml' #cambiar esto

with open(yml_file, "r") as ymlfile:
    full_cfg = yaml.safe_load(ymlfile)
    
cfg = Box({**full_cfg}, default_box=True, default_box_attr=None)
