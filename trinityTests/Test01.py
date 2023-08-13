import pandas as pd 
file_path =  ""

df = pd.read_csv(file_path)


import pubchempy as pq
from pubchemprops.pubchemprops import get_cid_by_name
from pubchemprops.pubchemprops import get_first_layer_props
from pubchemprops.pubchemprops import get_second_layer_props

easy_second = get_second_layer_props('acetone', ['Heat of Combustion'])

easy_properties = get_first_layer_props('acetone', ['MolecularWeight', 'IUPACName', 'CanonicalSMILES', 'InChI'])



heat_data = easy_second['Heat of Combustion'][0]['Value']['StringWithMarkup'][0]['String']
heat_data.split(': ')[1]


easy_properties = get_first_layer_props('acetone', ['MolecularWeight', 'IUPACName', 'CanonicalSMILES', 'InChI'])

print(easy_properties)

