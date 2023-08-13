from pubchemprops.pubchemprops import get_cid_by_name
from pubchemprops.pubchemprops import get_first_layer_props
from pubchemprops.pubchemprops import get_second_layer_props
import pandas as pd
import os
import json
import pubchempy as pcp
import urllib.parse
import re

dados = pd.read_csv(r"")
results = []
for compound in dados.cmpdname:
    try:
        encoded_compound = urllib.parse.quote(compound)
        print(compound)
        props = get_second_layer_props(encoded_compound, ['Heat of Combustion'])
        results.append({compound: props['Heat of Combustion'][0]['Value']['StringWithMarkup'][0]['String']})
        print(results)
    except Exception as e:
        print(f"An error occurred for compound '{compound}': {str(e)}")
        continue 


print(results)

res = pcp.get_compounds('Glucose', 'name')
for compound in res:
    print(compound.isomeric_smiles)


def extract_element_names(compound):
    element_names = re.findall(r'[A-Z][a-z]*', compound)
    return element_names


def is_list_present(element_names_list):
    return all(element in element_names_list for element in elements_and_smiles.keys())


data = []

for entry in results:
    for compound, value in entry.items():
        energy_match = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', value)
        if energy_match:
            energy = float(energy_match[-1])
            data.append([compound, energy])

df = pd.DataFrame(data, columns=['Compound', 'Energy'])

df['Element_Names'] = df['Compound'].apply(extract_element_names)

df['Elements_Present'] = df['Element_Names'].apply(is_list_present)

filtered_df = df[df['Elements_Present']]
filtered_df['SMILES'] = filtered_df['Compound'].apply(get_smiles)
filtered_df.drop(columns=['Element_Names', 'Elements_Present'], inplace=True)

df = df.merge(filtered_df[['Compound', 'SMILES']], on='Compound', how='left')

lista = []
for compounds in df.Compound:
    resultados = pcp.get_compounds(compounds, 'name')
    for results in resultados:
        composto = results.isomeric_smiles
        print("smiles: ", composto)
        print("name: ", compounds)
        lista.append([compounds,composto])


compound_smiles_dict = {item[0]: item[1] for item in lista}
df['SMILES'] = df['Compound'].map(compound_smiles_dict)
df.to_csv('compounds_with_smiles.csv', index=False)