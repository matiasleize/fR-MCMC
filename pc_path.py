import socket
def definir_path():
    if socket.gethostname() == 'matias-Inspiron-5570':
        path_git = '/home/matias/Documents/Tesis/fR-MCMC'
        path_datos_global = '/home/matias/Documents/Tesis'
    elif socket.gethostname() == 'quipus': #Compu de Susana
        path_git = '/home/mleize/fR-MCMC'
        path_datos_global = '/home/mleize'
    elif socket.gethostname() == 'mn328': #Cluster
        path_git = '/tmpu/dsy_g/mleiz_a/fR-MCMC'
        path_datos_global = '/tmpu/dsy_g/mleiz_a'
    else:
        path_git = 'mleize/fR-MCMC'
        path_datos_global = 'mleize'
    return path_git, path_datos_global


if __name__ == '__main__':
    #Much better:
    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    path_datos_global = os.path.dirname(path_git)

    print(path_git)
    print(path_datos_global)

    #For some reason this work:
    os.chdir(path_git)
    os.sys.path.append('./Software/Funcionales/')

    #And this not:
    os.chdir(path_git+'/Software/Funcionales/')


    #The header of all run files should have this format:
    import numpy as np; np.random.seed(42)
    import emcee

    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    os.chdir(path_git); os.sys.path.append('./Software/Funcionales/')
