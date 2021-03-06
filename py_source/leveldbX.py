'''
Created on Nov 25, 2015

@author: tim
'''
import leveldb
import simplejson
from Queue import Queue
from os.path import join, exists
import urllib2
from threading import Thread
from os.path import expanduser
from dateutil import parser
import os
import numpy as np
import h5py
home = expanduser("~")

def save_hdf5_matrix(filename,x):    
    file = h5py.File(filename,'w')
    file.create_dataset("Default", data=x)        
    file.close()
    
def load_hdf5_matrix(filename):    
    f = h5py.File(filename,'r')
    
    z = f['Default'][:]
    
    f.close()
    return z


class Table(object):
    def __init__(self, name, path, address, hasServer):
        self.name = name
        self.queue = Queue()
        self.hasServer = hasServer
        self.db = leveldb.LevelDB(join(path,name))
        self.address = address
        self.hdf5_path = join(path,name,'hdf5')
        if not exists(self.hdf5_path): os.makedirs(self.hdf5_path)
        
    def get(self, key):
        ret = None
        try:
            ret = self.db.Get(key)
        except KeyError:
            ret = None
            
        if self.hasServer and ret == None: 
            r = urllib2.urlopen(join(self.address, self.name, key))
            value = simplejson.load(r)
            self.set(key, value)
            return value
        else:
            if ret == None: return None
            if ret == "hdf5":
                return load_hdf5_matrix(join(self.hdf5_path, key.replace('/','_')+ ".hdf5"))
            else:
                return simplejson.loads(ret)
        
    
    def set(self, key, value):
        if isinstance(value, np.ndarray):     
            self.db.Put(key,"hdf5")
            save_hdf5_matrix(join(self.hdf5_path, key.replace('/','_')+ ".hdf5"), value)
        else:
            self.db.Put(key,simplejson.dumps(value))
        
    def post(self, key, value):
        if self.hasServer:
            req = urllib2.Request(join(self.address, self.name, key))
            req.add_header('Content-Type', 'application/json')
            print simplejson.dumps(value)
            if isinstance(value, basestring):
                response = urllib2.urlopen(req, value)
            else:
                response = urllib2.urlopen(req, simplejson.dumps(value))
            
            self.set(key, value)
        else:
            print 'Server not reachable!'
        
        
    def delete(self, key):
        self.db.Delete(join(self.name,key))
        
        
    def scan(self):
        for key, value in self.db.RangeIter():
            yield key, simplejson.loads(value)
        
        


class LevelDBX(object):
    def __init__(self, isServer = False, path=join(home, '.nlpdb'), ip='86.119.32.220', port=5000):
        self.isServer = isServer
        self.ip = ip
        self.port = 5000
        self.address = "http://{0}:{1}".format(ip,port)
        
        if isServer:
            self.path = join(home,'.nlpdb_server')     
        else:
            self.path  = path
            
        self.db = leveldb.LevelDB(self.path)
        
        try:
            stamp = parser.parse(simplejson.load(urllib2.urlopen(join( self.address, 'ping'))))
        except:
            stamp = None
        if stamp != None: self.hasServer = True
        else: self.hasServer = False
        
    def join(self):
        if self.isServer:
            self.bgServer.join()
        
    def get_table(self, name):
        return Table(name, self.path,  self.address, self.hasServer)
    
    def table_exists(self, name):
        return exists(join(self.path, name))




