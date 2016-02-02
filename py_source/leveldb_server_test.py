'''
Created on 27 Dec 2015

@author: timdettmers
'''
from leveldbX import LevelDBX
from threading import Thread





db = LevelDBX(path="/home/tim/test",isServer=False, ip='86.119.32.220')
test = db.get_table('test')

#test.post('abc','uden')
print test.get('abc')






