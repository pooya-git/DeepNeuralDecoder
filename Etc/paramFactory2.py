# ------------------------------------------------------------------------------
# 
#    This small script is a template generator for param.json of CNOTExRec
#
#    Copyright (C) 2017 Pooya Ronagh
# 
# ------------------------------------------------------------------------------

import json
from pprint import pprint

SYNBITS= 3
ERRBITS= 2**7

dic= {}
dic['Syn1']= {'dim': SYNBITS, \
	'edges': {'H1': {'w_std': 0.01, 'b_std':0.0}}}
dic['Syn2']= {'dim': SYNBITS, \
	'edges': {'H2': {'w_std': 0.01, 'b_std':0.0}}}
dic['H1']= {'dim': 10, \
	'edges': {'Err1': {'w_std': 0.316, 'b_std':0.0}, \
			  'HCNOT': {'w_std': 0.316, 'b_std':0.0}}}
dic['H2']= {'dim': 10, \
	'edges': {'Err2': {'w_std': 0.316, 'b_std':0.0}, \
			  'HCNOT': {'w_std': 0.316, 'b_std':0.0}}}
dic['HCNOT']= {'dim': 10, \
	'edges': {'H3': {'w_std': 0.316, 'b_std':0.0}, \
			  'H4': {'w_std': 0.316, 'b_std':0.0}}}
dic['Syn3']= {'dim': SYNBITS, \
	'edges': {'H3': {'w_std': 0.01, 'b_std':0.0}}}
dic['Syn4']= {'dim': SYNBITS, \
	'edges': {'H4': {'w_std': 0.01, 'b_std':0.0}}}
dic['H3']= {'dim': 10, \
	'edges': {'Err3': {'w_std': 0.316, 'b_std':0.0}}}
dic['H4']= {'dim': 10, \
	'edges': {'Err4': {'w_std': 0.316, 'b_std':0.0}}}

dic['Err1']= {'dim': ERRBITS}
dic['Err2']= {'dim': ERRBITS}
dic['Err3']= {'dim': ERRBITS}
dic['Err4']= {'dim': ERRBITS}

dic['Syn1']['type']= 'input'
dic['Syn2']['type']= 'input'
dic['Syn3']['type']= 'input'
dic['Syn4']['type']= 'input'
dic['H1']['type']= 'hidden'
dic['H2']['type']= 'hidden'
dic['HCNOT']['type']= 'hidden'
dic['H3']['type']= 'hidden'
dic['H4']['type']= 'hidden'
dic['Err1']['type']= 'output'
dic['Err2']['type']= 'output'
dic['Err3']['type']= 'output'
dic['Err4']['type']= 'output'

dictionaryToJson = json.dumps(dic, indent=2)
print(dictionaryToJson)
