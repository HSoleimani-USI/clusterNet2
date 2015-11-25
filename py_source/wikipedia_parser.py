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
rshortSummary = re.compile("(?:''')(.*)")

rHeaders = re.compile("(?:={2,4})(\w*?)(={2,4}\s)")

pages = []
i = 0
titels = []


with open(wiki_xml_path,'r') as f:
    
    page = []
    headers = []
    start = False
    end = False
    for line in f:
        if '<page>' in line: 
            start = True        
            page = []
            headers = []
        
        if start: page.append(line)
        if '</page>' in line:   
            i+=1       
            title =  rTitle.search(page[1]).group(1)            
            page = "".join(page)
            if len(page) < 1000: 
                continue            
            matches = rHeaders.findall(page)            
            if matches:
                for match in matches:
                    if len(match[0]) == 0: continue
                    header = match[0]
                    level = len(match[1])-1
                    headers.append((level, header))
            #match = rSummary.search(page)
            #if match: print match.group(1)
            
            
            match = rshortSummary.search(page)
            
            page_data = {'raw' : page, 'headers' : headers}
            if match:
                page_data['short_summary'] = match.group(1)
            
            raw.set(title, page_data)
            
            if i % 10000 == 0: print i
            


        







#

