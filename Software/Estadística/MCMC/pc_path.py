import socket
def definir_path(): 
    if socket.gethostname() == 'matias-Inspiron-5570':
        path_git = '/home/matias/Documents/Tesis/tesis_licenciatura'
        path_datos_global = '/home/matias/Documents/Tesis'
    elif: socket.gethostname() == 'quipus': #Compu de Susana
        path_git = '/home/mleize/tesis_licenciatura'
        path_datos_global = '/home/mleize'
    else: 
	path_git = '/home/mleize/tesis_licenciatura'
	path_datos_global = '/home/mleize'
    return path_git, path_datos_global
