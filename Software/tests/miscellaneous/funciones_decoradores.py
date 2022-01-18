import time
def timeit(method):
	def timed(*args, **kw):
		ts = time.time()
		result = method(*args, **kw)
		te = time.time()

		if 'log_time' in kw:
			name = kw.get('log_name', method.__name__.upper())
			kw['log_time'][name] = int((te - ts) * 1000)
		else:
			print('%r  %2.2f ms' % \
			(method.__name__, (te - ts) * 1000))
		return result

	return timed

#%%
if __name__ == '__main__':
	@timeit
	def lalala(a,b):
		for i in range(a):
			for j in range(b):
				aux_1 = i
				aux_2 = j
		print(aux_1+aux_2)

	lalala(1000,200000)
