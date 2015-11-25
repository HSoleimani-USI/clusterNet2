'''
Created on Nov 25, 2015

@author: tim
'''
import leveldb
import simplejson
from os.path import join

class Table(object):
    def __init__(self, name, path):
        self.name = name
        self.db = leveldb.LevelDB(join(path,name))
        
    def get(self, key):
        return simplejson.loads(self.db.Get(join(self.name, key)))
    
    def set(self, key, value):
        self.db.Put(join(self.name,key),simplejson.dumps(value))
        
        
    def delete(self, key):
        self.db.Delete(join(self.name,key))
        
        
    def scan(self):
        for key, value in self.db.RangeIter():
            yield key, simplejson.loads(value)
        
        


class LevelDBX(object):
    def __init__(self, path):
        self.db = leveldb.LevelDB(path)
        self.path = path        
        
    def get_table(self, name):
        return Table(name, self.path)

