"""
This file queries an existing filled NeuroKB instance loaded on neo4j
to create a HeteroData object to be saved in the data directory. This
object will be used later in the link prediction script.
"""

###
# IMPORTS
###

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import os
import re
from db.database import NeuroKBConnection
from rdkit import Chem
from rdkit.Chem import MACCSkeys

project_path = os.path.dirname(os.getcwd())

###
# SET UP DRIVER
###

neuroKB = NeuroKBConnection()

# creates the query text to gather all nodes and the corresponding
#     attributes of a provided node label
def get_node_query(label):
    with neuroKB.driver.session() as session:
        query = f"""MATCH (n:{label})
            WHERE n.id =2
            RETURN keys(n) AS propertyNames
            LIMIT 1;"""
        result = session.run(query)
        data = [record.data() for record in result]
        data = data[0]['propertyNames']

    key_query = ""
    data = [item for item in data if item != 'id']
    for key in data:
        key_query += f", n.{key} AS {key}"

    return f"""MATCH (n:{label}) RETURN n.id AS id{key_query};"""


# creates the query text to gather all edges of a provided edge label.
#     NeuroKB edges do not currently have edge properties so only the
#     source and sink node IDs are returned.
def get_edge_query(label):
    with neuroKB.driver.session() as session:
        query = f"""MATCH ()-[r:{label}]->()
            RETURN keys(r) AS propertyNames
            LIMIT 1;"""
        result = session.run(query)
        data = [record.data() for record in result]
        data = data[0]['propertyNames']

    final_query = ""
    if len(data)==0:
        final_query = f"""MATCH ()-[r:{label}]->()
                          RETURN type(r) AS relationshipType, startNode(r).id AS startNode, 
                          endNode(r).id AS endNode;"""
    else:
        print(f"Edge {label} has properties")
    return final_query

###
# INITIALIZE HETEROGRAPH
###

data = HeteroData()

###
# GET NODE DATA
###

node_labels = ["Drug", "Gene", "DrugClass", "BodyPart", "BiologicalProcess",
               "MolecularFunction", "CellularComponent", "Pathway", "Disease"]

drug_id_key = {}

print("BEGIN NODE CONSTRUCTION")
for n in node_labels:
    print(f" > BUILDING {n} NODE")

    nodes = neuroKB.query_to_dataframe(get_node_query(n))

    # convert to numeric data
    features = nodes.iloc[:, 1:]
    feature_names = features.columns.tolist()

    if n == "Gene":
        ids = torch.tensor(nodes.iloc[:, 0].values, dtype=torch.float32)
        data[n].node_id = ids

        # TODO - we may want to keep Gene NaN values but for now we are replacing them with zeros
        features.fillna(0, inplace=True)

        chromosomes = []
        chromosome_arms = []
        major_bands = []
        sub_bands = []
        for index, row in features.iterrows():
            chromosome = row['chromosome']
            loc = str(row['location'])

            if chromosome == "X":
                chromosome = 23
            if chromosome == "Y":
                chromosome = 24
            if chromosome == "X|Y":
                chromosome = 23.5
            if chromosome == "-" or chromosome == "Un":
                chromosome = 0
            if chromosome == "MT":
                chromosome = 25

            loc = loc.replace('X', '23')
            loc = loc.replace('Y', '24')

            chromosome_arm_key = {'p':0, 'q':1, 'pter':2, 'qter':3, 'cen':4,
                                  'pter-q':5, 'cen-q':6, 'pter-p':7}

            if loc != "-" and loc != "0" and loc != "23;24" and pd.notna(loc):
                simple_pattern = re.match(r"^(\d+)([pq])$", loc)
                standard_pattern = re.match(r"(\d+)([pq])(\d+)", loc)
                last_pattern = re.match(r"^(\d+)([^\d]+)(\d+(\.\d+)?)$", loc)

                if simple_pattern:
                    major_band = simple_pattern.group(1)
                    chromosome_arm = chromosome_arm_key[simple_pattern.group(2)]
                    sub_band = 0

                elif standard_pattern:
                    major_band = standard_pattern.group(1)
                    chromosome_arm = chromosome_arm_key[standard_pattern.group(2)]
                    sub_band = standard_pattern.group(3)

                elif last_pattern:
                    major_band = last_pattern.group(1)
                    chromosome_arm = chromosome_arm_key[last_pattern.group(2)]
                    sub_band = last_pattern.group(3)
                else:
                    # one case that is not caught = 13cen
                    other = re.match(r"^(\d{1,2})(.+)$", loc)
                    major_band = other.group(1)
                    chromosome_arm = chromosome_arm_key[other.group(2)]
                    sub_band = 0
            else:
                major_band = 0
                chromosome_arm = 0
                sub_band = 0

            chromosomes.append(float(chromosome))
            chromosome_arms.append(float(chromosome_arm))
            major_bands.append(float(major_band))
            sub_bands.append(float(sub_band))

        f = torch.tensor([chromosomes, chromosome_arms, major_bands, sub_bands]).T
    elif n == "Drug":
        # Drug data is mostly numeric so we mostly just need to convert the features into a tensor
        # TODO - right now we are removing alogs because negative values are meaningful
        #        if there is another way to impute NaNs later we will have to add that in
        attr = nodes.drop(['commonName','alogs', 'smiles'], axis=1)

        # fill the node_id heterodata attribute with the new ids
        ids = torch.tensor(attr['id'].values, dtype=torch.float32)
        data[n].node_id = ids

        # drop the id attribute because we don't need it now
        attr = attr.drop(['id'], axis=1)

        # enhanced stereo is a boolean so we replace F with 0 and T with 1 and keep the NaN
        attr['enhanced_stereo'] = attr['enhanced_stereo'].apply(lambda x: int(x) if pd.notna(x) else x)
        attr.fillna(attr.mean(), inplace=True)        # replace any NaNs with the mean of the column
        f = torch.tensor(attr.values, dtype=torch.float32)

        # TODO - if you want to use MACCS keys for the drug definitions, uncomment this code
        #    and comment out the section above from line 169. Performance did not
        #    drastically improve beyond the current Drug attributes.
        # all_maccskeys = []
        # errors = []
        # for i, row in nodes.iterrows():
        #     smiles = row['smiles']
        #     m = Chem.MolFromSmiles(smiles)
        #     try:
        #         MACCskeys = list(MACCSkeys.GenMACCSKeys(m))
        #         all_maccskeys.append(MACCskeys)
        #     except Exception as e:
        #         sub = [0] * 167
        #         errors.append(i)
        #         all_maccskeys.append(sub)

        # ids = torch.tensor(nodes['id'].values, dtype=torch.float32)
        # data[n].node_id = ids
        # f = torch.tensor(all_maccskeys, dtype=torch.float32)

        # add the MACCSkey and the existing drug node attributes to get all the drug nodes
        # all_drug_attr = []
        # for ndx, row in attr.iterrows():
        #     full_row = row.to_list() + all_maccskeys[ndx]
        #     all_drug_attr.append(full_row)
        # f = torch.tensor(all_drug_attr, dtype=torch.float32)
    else:
        ids = torch.tensor(nodes.iloc[:, 0].values, dtype=torch.float32)
        data[n].node_id = ids

        # if it is any other node, just use a zero vector of length n because we aren't using
        #     any of those features
        f = torch.zeros(len(features.values)).unsqueeze(1)

    data[n].x = f

###
# GET EDGE DATA
###

edge_labels = [["Drug", "CHEMICALBINDSGENE", "Gene"],
               ["Drug", "CHEMICALINCREASESEXPRESSION", "Gene"],
               ["Drug", "CHEMICALDECREASESEXPRESSION", "Gene"],
               ["Drug", "DRUGINCLASS", "DrugClass"],
               ["Drug", "DRUGINDICATIONDISEASE", "Disease"],
               ["Drug", "DRUGOFFLABELUSEDISEASE", "Disease"],
               ["Drug", "DRUGCONTRAINDICATIONDISEASE", "Disease"],
               ["Gene", "GENEPARTICIPATESINBIOLOGICALPROCESS", "BiologicalProcess"],
               ["Gene", "GENEREGULATESGENE", "Gene"],
               ["Gene", "GENEINPATHWAY", "Pathway"],
               ["Gene", "GENEINTERACTSWITHGENE", "Gene"],
               ["Gene", "GENEHASMOLECULARFUNCTION", "MolecularFunction"],
               ["Gene", "GENECOVARIESWITHGENE", "Gene"],
               ["Gene", "GENEASSOCIATEDWITHCELLULARCOMPONENT", "CellularComponent"],
               ["Gene", "GENEASSOCIATESWITHDISEASE", "Disease"],
               ["BodyPart", "BODYPARTOVEREXPRESSESGENE", "Gene"],
               ["BodyPart", "BODYPARTUNDEREXPRESSESGENE", "Gene"]]

print("\nBEGIN NODE CONSTRUCTION")
for e in edge_labels:
    print(f" > BUILDING {e} EDGES")
    edges = neuroKB.query_to_dataframe(get_edge_query(e[1]))
    edge_index = torch.tensor(edges[['startNode', 'endNode']].values.T, dtype=torch.long)
    data[e[0], e[1], e[2]].edge_index = edge_index

###
# SAVE HETERODATA
###

print(data)
torch.save(data, f'{project_path}/data/neuroKB.pth')

###
# CLOSE DRIVER
###

neuroKB.close()