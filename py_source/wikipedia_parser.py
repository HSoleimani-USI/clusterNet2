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
import lucene
import cPickle as pickle
import time
import numpy as np
from WikiExtractor import Extractor
import codecs







DBpath = sys.argv[1]
wiki_xml_path = sys.argv[2]


'''
lucene.initVM()
indexDir = SimpleFSDirectory(File(DBpath+"/lucene/index/"))
writerConfig = IndexWriterConfig(Version.LUCENE_CURRENT, StandardAnalyzer(Version.LUCENE_CURRENT))
writer = IndexWriter(indexDir, writerConfig)

print "%d docs in index" % writer.numDocs()
print "Reading lines from sys.stdin..."
'''


print DBpath

db = leveldbX.LevelDBX(path=DBpath)

raw = db.get_table('raw_pages')


rTitle = re.compile(".*<title>(.*?)</title>")
rshortSummary = re.compile("(?:''')(.*)")


rSummary = re.compile("(?:''')(.*)(?:={2,5})")

rHeaders = re.compile("(?:={2,4})(\w*?)(={2,4}\s)")

rLinks = re.compile("(?:\[\[)(.*?)(?:]])")

pages = []
i = 0
titels = []

graph = {}


'''
with open(wiki_xml_path,'r') as f:
    for i, line in enumerate(f):
        pass
    print "total lines: ", i
'''
total_lines = 833379424


t0 = time.time()
with open(wiki_xml_path,'r') as f:
    
    page = []
    headers = []
    start = False
    end = False
    for lineno, line in enumerate(f):
        if '<page>' in line:
            start = True        
            page = []
            headers = []
            links = []
        
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
            
            
            #print page          
            graph[title] = {}  
            matches = rLinks.findall(page)
            if matches:
                for match in matches:                                        
                    link = match.split('|')
                    if len(link) > 1:
                        article = link[1]
                        link = link[0]
                        graph[title][article] = 1
                    else:
                        link = link[0]
                    links.append(link.lower().replace('[','').replace(']',''))
                    
            
            
            
            match = rshortSummary.search(page)
            
            #print page
            
            content = page.decode('utf-8').encode('ascii', 'ignore')
            lines = Extractor(0, title, content).extract()
                
            
            page = "".join(lines)
            page_data = {'raw' : page, 'headers' : headers, 'links' : links}
            if match:
                page_data['short_summary'] = match.group(1)
                            
            raw.set(title.lower(), page_data)
            i+=1
            
            '''            
            doc = Document()
            doc.add(Field("page", page, Field.Store.YES, Field.Index.ANALYZED))
            writer.addDocument(doc)
            '''
            if i % 10000 == 0: 
                print i
                #print "%d docs in index" % writer.numDocs()
                
        
        if lineno % 1000000 == 0: 
            lines_per_sec = (lineno+1)/(time.time()-t0)
            print "{0} lines per second; ETA: {1}h".format(int(lines_per_sec), np.round(((total_lines-lineno)/lines_per_sec)/60./60.,2))
                #print "%d docs in index" % writer.numDocs()
            


        
#writer.close()
