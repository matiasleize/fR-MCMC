import getdist

a = getdist.mcsamples.loadMCSamples('/home/matias/Documents/Tesis/tesis_licenciatura/chains/first_run')

params_names = [a.parName(i) for i in range(len(a.samples[0,:]))]

a.PCA(params_names)

#For the first parameter
a.confidence(1, 0.025)
a.confidence(1, 0.025,upper=True)

