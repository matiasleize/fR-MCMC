import socket
def definir_path(): 
    if socket.gethostname() == 'matias-Satellite-A665':
        path_git = '/home/Documents/Tesis/tesis_licenciatura'
        path_datos_global = '/home/Documents/Tesis'
    else: #socket.gethostname() == 'compu de Susana':
        path_git = '/home/mleize/tesis_licenciatura'
        path_datos_global = '/home/mleize'
    return path_git, path_datos_global
