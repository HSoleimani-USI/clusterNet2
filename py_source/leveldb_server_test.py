'''
Created on 27 Dec 2015

@author: timdettmers
'''
from leveldbX import LevelDBX
from threading import Thread





db = LevelDBX(isServer=False)
test = db.get_table('test')

test.post('abc',{"afgdsf" : 5346.33})
print test.get('abc')






