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
dic['Syn12']= {'dim': SYNBITS+SYNBITS, \
	'edges': {'H': {'w_std': 0.01, 'b_std':0.0}}}
dic['H']= {'dim': 30, \
	'edges': {'Err3': {'w_std': 0.18257, 'b_std':0.0}, \
			  'Err4': {'w_std': 0.18257, 'b_std':0.0}}}
dic['Syn3']= {'dim': SYNBITS, \
	'edges': {'Err3': {'w_std': 0.01, 'b_std':0.0}}}
dic['Syn4']= {'dim': SYNBITS, \
	'edges': {'Err4': {'w_std': 0.01, 'b_std':0.0}}}
dic['Err3']= {'dim': ERRBITS}
dic['Err4']= {'dim': ERRBITS}

dic['Syn12']['type']= 'input'
dic['Syn3']['type']= 'input'
dic['Syn4']['type']= 'input'
dic['H']['type']= 'hidden'
dic['Err3']['type']= 'output'
dic['Err4']['type']= 'output'

dictionaryToJson = json.dumps(dic, indent=2)
print(dictionaryToJson)
