from pubchemprops.pubchemprops import get_cid_by_name
from pubchemprops.pubchemprops import get_first_layer_props
from pubchemprops.pubchemprops import get_second_layer_props
import pandas as pd
import os
import json
import pubchempy as pcp


compound = pcp.Compound.from_cid(5090)

easy_second = get_second_layer_props('1,2-dichloroethane', ['Heat of Combustion'])

dados = pd.read_csv(r"")
c1 = dados.cmpdname[0]

easy_second = get_second_layer_props(c1, ['Heat of Combustion'])

