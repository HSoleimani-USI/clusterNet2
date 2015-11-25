#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Nov 24, 2015

@author: tim
'''
import mistune
import re
import leveldbX
import sys


DBpath = sys.argv[1]
wiki_xml_path = sys.argv[2]




db = leveldbX.LevelDBX(DBpath)

raw = db.get_table('raw_pages')


rTitle = re.compile(".*<title>(.*?)</title>")
rSummary = re.compile(".*}}(.*?)==")

rHeaders = re.compile("(?:={2,4})(\w*?)(={2,4}\s)")

pages = []
i = 0
titels = []

with open(wiki_xml_path,'r') as f:
    
    page = []
    start = False
    end = False
    for line in f:
        if '<page>' in line: start = True
        
        if start: page.append(line)
        if '</page>' in line:    
            
            title =  rTitle.search(page[1]).group(1)
            
            
            i+=1      
            page = "".join(page)
            headers = rHeaders.search(page)
            if headers:
                header = headers.group(1)
                level = len(headers.group(2))
            #match = rSummary.search(page)
            #if match: print match.group(1)
            
            #print page
            raw.set(title, page)
            
            page = []
            if i % 10000 == 0: print i
            


        







#

