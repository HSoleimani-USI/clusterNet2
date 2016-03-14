import library_interface as lib
import numpy as np
import ctypes as ct
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
import string
from os.path import join,isfile
import cPickle as pickle
import time
from os import listdir
import nltk
import h5py
import leveldbX



class TextToIndex(object):
	def __init__(self, name, path, output_dir):
		self.vocab2freq = {}
		self.vocab2idx = {}		
		self.path = path
		self.output_dir = output_dir
		self.db = leveldbX.LevelDBX()
		self.tbl = self.db.get_table('vocab')
		self.name = name
		if isfile(path): self.files = [self.path]
		else:
			self.files =  [join(path,f) for f in listdir(path) if isfile(join(path, f))]
		
	def create_vocabulary(self, filter_stopwords=True, save_to_disk=True):
		tokenizer = WordPunctTokenizer()
		try:			
			stop = stopwords.words('english') + list(string.punctuation)
		except LookupError:
			print "Stopwords not found! Please download them via the nltk interface and try again"
			nltk.download()
			exit()
		t0 = time.time()
		current_idx = 0
		for path in self.files:
			with open(path) as f:
				for lineno, line in enumerate(f):
					if lineno % 10000 == 0:
						if lineno > 0:
							print "Current line number is {0}. Operating at {1} lines per second".format(lineno, int(lineno/(time.time()-t0)))
						
					if filter_stopwords:
						words = [word for word in tokenizer.tokenize(line.lower()) if word not in stop]
					else:
						words = [word for word in tokenizer.tokenize(line.lower())]
						
					for word in words:
						if word not in self.vocab2freq: 
							self.vocab2freq[word] = 0
							self.vocab2idx[word] = current_idx
							current_idx+=1
						self.vocab2freq[word] +=1
			
		self.tbl.set(join(self.name,'vocab2freq'), self.vocab2freq)
		self.tbl.set(join(self.name,'vocab2idx'), self.vocab2idx)		
		if save_to_disk:
			pickle.dump(self.vocab2freq, open(join(self.output_dir, 'vocab2freq.p'),'wb'), pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.vocab2idx, open(join(self.output_dir, 'vocab2idx.p'),'wb'), pickle.HIGHEST_PROTOCOL)
			
	def create_idx_files(self, save_to_disk=True):
		tokenizer = WordPunctTokenizer()
		try:			
			sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		except LookupError:
			print "English punct tokenizer not found! Please download them via the nltk interface and try again"
			nltk.download()
			exit()
		t0 = time.time()
		corpa = []
		sent_idx = []
		max_length = 0
		i = 0
		print 'parsing sentences...'
		for path in self.files:
			with open(path) as f:
				corpa.append(f.readlines())
		for buffer in corpa:		
			text = "".join(buffer)					
			sentences = sent_detector.tokenize(text.lower().strip())
			for sent in sentences:
				i+=1
				if i % 10000 == 0:
					if i > 0:
						print "Sencente number: {0}. Operating at {1} sentences per second".format(i, int(i/(time.time()-t0)))
				words = [word for word in tokenizer.tokenize(sent)]
				
				idx = []
				for word in words:
					if word not in self.vocab2idx: continue
					idx.append(self.vocab2idx[word])
				if len(idx) > max_length: 
					max_length  = len(idx)
					#print max_length, sent
				sent_idx.append(idx)
				
		data = np.ones((i,max_length),np.int32)*-1
		print data.shape		
		for sentno, idx_values in enumerate(sent_idx):
			for idxno, idx in enumerate(idx_values):
				data[sentno,idxno] = idx
				
		self.tbl.set(join(self.name,'idx'),data)
		if save_to_disk:
			save_hdf5_matrix(join(self.output_dir,'idx.hdf5'), data)
			
					
				
				
					
		
		

class NeuralNetwork(object):
	
	def __init__(self, X, y, layers = [1024, 1024], classes = 10):
		cv_size = 0.2
		n = X.shape[0]
		cv_start = n*(1.0-cv_size)
		x_cv = X[cv_start:].copy()
		y_cv = y[cv_start:].copy()
		layers = np.array(layers, dtype=np.float32)
		train = CPUBatchAllocator(X[:cv_start], y[:cv_start], 128)
		cv = CPUBatchAllocator(x_cv, y_cv, 128)
		self.net_pt = lib.funcs.fget_neural_net(lib.pt_clusterNet, train.pt, cv.pt, layers.ctypes.data_as(ct.POINTER(ct.c_float)),layers.shape[0], 1, classes)
		
		'''
		pt_train = lib.funcs.fget_CPUBatchAllocator(
					X[:cv_start].ctypes.data_as(ct.POINTER(ct.c_float)),
					y[:cv_start].ctypes.data_as(ct.POINTER(ct.c_float)),
					int(X[:cv_start].shape[0]), int(X.shape[1]), int(y.shape[1]),
					int(128))
		
		
		pt_cv = lib.funcs.fget_CPUBatchAllocator(
					x_cv.ctypes.data_as(ct.POINTER(ct.c_float)),
					y_cv.ctypes.data_as(ct.POINTER(ct.c_float)),
					int(x_cv.shape[0]), int(X.shape[1]), int(y.shape[1]),
					int(128))
		
		self.net_pt = lib.funcs.fget_neural_net(lib.pt_clusterNet, pt_train, pt_cv, layers.ctypes.data_as(ct.POINTER(ct.c_float)),layers.shape[0], 1, classes)
		'''
		
	def fit(self):
		lib.funcs.ffit_neural_net(self.net_pt)

class Timer(object):
	def __init__(self):
		self.pt = lib.funcs.fget_Timer()
		
	def tick(self, name='default'):
		lib.funcs.ftick(self.pt, ct.c_char_p(name))
		
	def tock(self, name='default'):		
		return lib.funcs.ftock(self.pt, ct.c_char_p(name))
	
class VectorSpace(object):
	def __init__(self, x, vocabulary=None, stopwords=None):
		self.rows = x.shape[0]
		self.dim = x.shape[1] 
		self.bufferT = array(x)
		self.vec = empty((self.dim,1))
		self.buffer = empty((self.dim,self.rows))
		self.X = self.bufferT.T	
		self.distances = empty((self.rows,1))
		self.distances_cpu = np.empty((self.rows,),dtype=np.float32)
		
		self.vocab2idx = {}
		self.idx2vocab = {}		
		self.stopdict = {}
		
		if stopwords != None:
			if isinstance(stopwords, dict): self.stopdict = stopwords
			else:
				for word in stopwords:
					self.stopdict[word] = 1
		
		for i, word in enumerate(vocabulary):
			word = word.strip().lower()
			self.vocab2idx[word] = i
			self.idx2vocab[i] = word
			
		
		
		
	def find_nearest(self, strValue, split=True, top=50):				
		vec = zeros((self.vec.shape[0], 1))
		slice_buffer = zeros((self.vec.shape[0], 1))	
			
		
		word_count = 0
		for word in strValue.strip().lower().split(' '):
			if word in self.vocab2idx and word not in self.stopdict:
				word_count +=1
				idx = self.vocab2idx[word]
				slice(self.X,0,self.dim,idx,idx+1,slice_buffer)
				add(vec,slice_buffer,vec)
				
				
		if word_count == 0:
			return [None, None]
		
		scalar_mul(vec,1.0/word_count,vec)				
		
		
		vector_sub(self.X, vec, self.buffer)
		pow(self.buffer, 2.0, self.buffer)
		transpose(self.buffer, self.bufferT)
		row_sum(self.bufferT, self.distances)
		sqrt(self.distances, self.distances)	
			
		tocpu(self.distances, self.distances_cpu)
		closest_idx = np.int32(np.argsort(self.distances_cpu)[0:top])
		
		closest_words = []
		for idx in closest_idx:
			closest_words.append(self.idx2vocab[idx])
			
		
		return [closest_idx, closest_words]
		
	

class BatchAllocator(object):
	def __init__(self, X, y, batch_size,buffertype='CPU'):
		self.current_batch = 0
		self.epoch = 0
		self.batches = X.shape[0]/batch_size
		if buffertype == 'CPU':
			self.pt = lib.funcs.fget_CPUBatchAllocator(lib.pt_clusterNet, 
						X.ctypes.data_as(ct.POINTER(ct.c_float)),
						y.ctypes.data_as(ct.POINTER(ct.c_float)),
						int(X.shape[0]), int(X.shape[1]), int(y.shape[1]),
						int(batch_size))
		elif buffertype == 'GPU':
			self.pt = lib.funcs.fget_GPUBatchAllocator(lib.pt_clusterNet, 
						X.ctypes.data_as(ct.POINTER(ct.c_float)),
						y.ctypes.data_as(ct.POINTER(ct.c_float)),
						int(X.shape[0]), int(X.shape[1]), int(y.shape[1]),
						int(batch_size))
		else:
			raise Exception("Batch allocator buffertype not supported. The supported types are: CPU, GPU")
		
		self.batchX = array(None, lib.funcs.fgetBatchX(self.pt), (batch_size, X.shape[1]))
		self.batchY = array(None, lib.funcs.fgetBatchY(self.pt), (batch_size, y.shape[1]))
		
	def alloc_next_async(self): lib.funcs.falloc_next_batch(self.pt)
	def replace_current_with_next_batch(self): 
		self.current_batch +=1
		if self.current_batch == self.batches: 			
			self.current_batch = 0
			self.epoch += 1
		lib.funcs.freplace_current_with_next_batch(self.pt)
	
	@property
	def X(self):		
		pt =  lib.funcs.fgetBatchX(self.pt)
		self.batchX.pt = pt
		return self.batchX
		
	@property
	def Y(self):		
		pt =  lib.funcs.fgetBatchY(self.pt)
		self.batchY.pt = pt
		return self.batchY


class array(object):
	def __init__(self, numpy_array=None, pt=None, shape=None):
		if pt != None: self.pt = pt
		if shape != None: 
			self.shape = shape
			if self.shape[1] == 1:
				self.shape = (shape[0],)		
		if numpy_array != None:
			if len(numpy_array.shape) > 2: raise Exception("Array must be one or two dimensional!")
			self.shape = numpy_array.shape
			if len(self.shape) == 1:
				self.pt = lib.funcs.fempty(lib.pt_clusterNet, numpy_array.shape[0], 1)
			else:
				self.pt = lib.funcs.fempty(lib.pt_clusterNet, numpy_array.shape[0], numpy_array.shape[1])
			lib.funcs.fto_gpu(lib.pt_clusterNet, numpy_array.ctypes.data_as(ct.POINTER(ct.c_float)), self.pt)
		
		self.cpu_arr = numpy_array
		


	def tocpu(self):
		if self.cpu_arr == None: self.cpu_arr = np.empty(self.shape, dtype=np.float32)
		lib.funcs.fto_host(lib.pt_clusterNet, self.pt,self.cpu_arr.ctypes.data_as(ct.POINTER(ct.c_float)))
		return self.cpu_arr
	
	@property
	def T(self): return array(None, lib.funcs.fT(lib.pt_clusterNet, self.pt), self.shape[::-1])

	

def save_hdf5_matrix(filename,x):    
	file = h5py.File(filename,'w')
	file.create_dataset("Default", data=x)        
	file.close()
	
def load_hdf5_matrix(filename):    
	f = h5py.File(filename,'r')
	
	z = f['Default'][:]
	
	f.close()
	return z
	
def ones(shape, dtype=np.float32):
	rows, cols = handle_shape(shape)
	return array(None, lib.funcs.ffill_matrix(lib.pt_clusterNet, rows,cols,ct.c_float(1.0)), shape)

def empty(shape, dtype=np.float32):
	rows, cols = handle_shape(shape)
	return array(None, lib.funcs.fempty(lib.pt_clusterNet, rows,cols), shape)


def setRandomState(seed): lib.funcs.fsetRandomState(lib.pt_clusterNet, seed)
def rand(rows, cols, dtype=np.float32):	return array(None, lib.funcs.frand(lib.pt_clusterNet,rows,cols),(rows, cols))
def randn(rows, cols, dtype=np.float32): return array(None, lib.funcs.frandn(lib.pt_clusterNet,rows,cols),(rows, cols))


def handle_shape(shape):
	if len(shape) == 1: return (shape[0],1)
	elif len(shape) > 2: raise Exception("Array must be one or two dimensional!")
	else: return shape
	
def dot(A,B,out=None):
	if not out: out = empty((A.shape[0],B.shape[1]))
	lib.funcs.fdot(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out


def abs(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.ffabs(lib.pt_clusterNet, A.pt, out.pt)
	return out

def log(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.flog(lib.pt_clusterNet, A.pt, out.pt)
	return out

def sqrt(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fsqrt(lib.pt_clusterNet, A.pt, out.pt)
	return out

def pow(A, power, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fpow(lib.pt_clusterNet, A.pt, out.pt, ct.c_float(power))
	return out

def logistic(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.flogistic(lib.pt_clusterNet, A.pt, out.pt)
	return out

def rectified_linear(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.frectified(lib.pt_clusterNet, A.pt, out.pt)
	return out

def logistic_grad(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.flogistic_grad(lib.pt_clusterNet, A.pt, out.pt)
	return out

def rectified_linear_grad(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.frectified_grad(lib.pt_clusterNet, A.pt, out.pt)
	return out

def copy(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fcopy(lib.pt_clusterNet, A.pt, out.pt)
	return out

def add(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fadd(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def sub(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fsub(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def div(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fdiv(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def mul(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fmul(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def equal(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.feq(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def less(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.flt(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def greater(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fgt(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def less_equal(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fle(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def greater_equal(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fge(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def not_equal(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fne(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def squared_difference(A, B, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fsquared_diff(lib.pt_clusterNet, A.pt, B.pt, out.pt)
	return out

def vector_add(A, v, out=None):	
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fvadd(lib.pt_clusterNet, A.pt, v.pt, out.pt)
	return out

def vector_sub(A, v, out=None):	
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fvsub(lib.pt_clusterNet, A.pt, v.pt, out.pt)
	return out

def create_t_matrix(v, max_value, out=None):		
	if not out: out = empty((v.shape[0],max_value+1))
	lib.funcs.ftmatrix(lib.pt_clusterNet, v.pt, out.pt)
	return out

def slice(A, rstart, rend, cstart, cend, out=None):
	if not out: out = empty((rend-rstart,cend-cstart))
	lib.funcs.fslice(lib.pt_clusterNet, A.pt, out.pt, rstart, rend, cstart, cend)
	return out

def softmax(A, out=None):
	if not out: out = empty((A.shape[0],A.shape[1]))
	lib.funcs.fsoftmax(lib.pt_clusterNet, A.pt, out.pt)
	return out

def argmax(A, out=None):
	if not out: out = empty((A.shape[0],1))
	lib.funcs.fargmax(lib.pt_clusterNet, A.pt, out.pt)
	return out

'''
def to_pinned(X):
	pt = lib.funcs.fto_pinned(lib.pt_clusterNet, X.shape[0], X.shape[1],
						X.ctypes.data_as(ct.POINTER(ct.c_float)))
	buffer = np.core.multiarray.int_asbuffer(ct.addressof(pt.contents), 4*X.size)
	return np.frombuffer(buffer, np.float32).reshape(X.shape)
'''
def row_sum(A, out=None):
	if not out: out = empty((A.shape[0],1))
	lib.funcs.frowSum(lib.pt_clusterNet, A.pt, out.pt)
	return out

def row_max(A, out=None):
	if not out: out = empty((A.shape[0],1))
	lib.funcs.frowMax(lib.pt_clusterNet, A.pt, out.pt)
	return out

def row_mean(A, out=None):
	if not out: out = empty((A.shape[0],1))
	lib.funcs.frowMean(lib.pt_clusterNet, A.pt, out.pt)
	return out

def col_sum(A, out=None):
	if not out: out = empty((A.shape[1],1))
	lib.funcs.fcolSum(lib.pt_clusterNet, A.pt, out.pt)
	return out

def col_max(A, out=None):
	if not out: out = empty((A.shape[1],1))
	lib.funcs.fcolMax(lib.pt_clusterNet, A.pt, out.pt)
	return out

def col_mean(A, out=None):
	if not out: out = empty((A.shape[1],1))
	lib.funcs.fcolMean(lib.pt_clusterNet, A.pt, out.pt)
	return out

def transpose(A, out=None):
	if not out: out = empty((A.shape[1],A.shape[0]))
	lib.funcs.ftranspose(lib.pt_clusterNet, A.pt, out.pt)
	return out

def max(A): return lib.funcs.ffmax(lib.pt_clusterNet, A.pt)
def sum(A): return lib.funcs.ffsum(lib.pt_clusterNet, A.pt)
def mean(A): return lib.funcs.ffmean(lib.pt_clusterNet, A.pt)

def get_view(A, rstart=0, rend=None):	
	if rend == None: rend = A.shape[0]
	return array(None, lib.funcs.fget_view(lib.pt_clusterNet, A.pt,rstart, rend), (rend-rstart, A.shape[1]))
	

	
def get_closest_index(x, top=50):
	rows = x.shape[0]
	dim = x.shape[1] 
	X = array(x)
	buffer = empty(x.shape)
	vec = empty((dim,1))
	distances = empty((rows,1))
	row_indexes = []
	distances_cpu = np.empty((rows,),dtype=np.float32)
	for i in range(x.shape[0]):
		if i % 100 == 0: print i
		slice(X, i,i+1,0, dim, vec)
		vector_sub(X, vec, buffer)
		pow(buffer, 2.0, buffer)
		row_sum(buffer, distances)
		sqrt(distances, distances)	
		
		tocpu(distances, distances_cpu)
		row_indexes.append(np.int32(np.argsort(distances_cpu)[:-top-1:-1]))
	return np.array(row_indexes)

def tocpu(A, out):
	lib.funcs.fto_host(lib.pt_clusterNet, A.pt,out.ctypes.data_as(ct.POINTER(ct.c_float)))
	
def printmat(A, rstart=None, rend=None, cstart=None,cend=None):
	if rstart and rend and cstart and cend: lib.funcs.fprintmat(lib.pt_clusterNet, A.pt, rstart, rend, cstart, cend)
	else: lib.funcs.fprintmat(lib.pt_clusterNet, A.pt, 0, int(A.shape[0]), 0, int(A.shape[1]))
	
def lookup_rowwise(embedding, idx_batch, out=None):
	if not out: out = empty((idx_batch.shape[0]*idx_batch.shape[1],embedding.shape[1]))
	lib.funcs.flookup(lib.pt_clusterNet, embedding.pt, idx_batch.pt, out.pt)
	return out
	

