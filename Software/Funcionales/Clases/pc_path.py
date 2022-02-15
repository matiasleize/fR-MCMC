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
