import socket
def definir_path():
    if socket.gethostname() == 'matias-Inspiron-5570':
        path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
        path_datos_global = '/home/matias/Documents/Tesis'
    elif socket.gethostname() == 'quipus': #Compu de Susana
        path_git = '/home/mleize/tesis_licenciatura'
        path_datos_global = '/home/mleize'
    elif socket.gethostname() == 'mn328': #Cluster
        path_git = '/tmpu/dsy_g/mleiz_a/tesis_licenciatura'
        path_datos_global = '/tmpu/dsy_g/mleiz_a'
    elif socket.gethostname() == 'ubuntu2104': #Compu casa Susana
        path_git = '/home/ubuntu/Documents/Tesis/tesis_licenciatura'
        path_datos_global = '/home/ubuntu/Documents/Tesis/'
    else:
        path_git = 'mleize/tesis_licenciatura'
        path_datos_global = 'mleize'
    return path_git, path_datos_global
