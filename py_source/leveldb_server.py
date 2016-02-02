'''
Created on Sep 29, 2014

@author: tim
'''
from flask import Flask, request, Response
from flask.ext.cors import CORS
import simplejson
from crossdomain import crossdomain
from os.path import expanduser
from datetime import datetime
from leveldbX import LevelDBX

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['db'] = LevelDBX(isServer=True)

tbls = {}

db = app.config['db'] 


@app.route('/ping', methods=['GET'])
@crossdomain(origin='*')
def ping():
    return Response(simplejson.dumps(str(datetime.utcnow())),  mimetype='application/json') 

@app.route('/<tbl>/<key>', methods=['GET'])
@crossdomain(origin='*')
def get(tbl, key): 
    if tbl not in tbls and not db.table_exists(tbl):
        return Response(simplejson.dumps('Table does not exist!'),  mimetype='application/json')  
    if tbl not in tbls: tbls[tbl] = db.get_table(tbl)
    
    return Response(tbls[tbl].get(key),  mimetype='application/json') 


@app.route('/<tbl>/<key>', methods=['POST'])
@crossdomain(origin='*')
def set(tbl, key): 
    if tbl not in tbls:
        tbls[tbl] = db.get_table(tbl)
        
    value = request.data
    tbls[tbl].set(key, value)
    
    return ""
    


app.run(debug = False, host="0.0.0.0")
    
    