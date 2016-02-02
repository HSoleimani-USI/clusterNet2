'''
Created on 27 Dec 2015

@author: timdettmers
'''
from leveldbX import LevelDBX
from threading import Thread





db = LevelDBX(path="/home/tim/test",isServer=False)
test = db.get_table('test')


print test.get('abc')






