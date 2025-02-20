import torch, csv, os, datetime, re
from model import Model, device, data
import pandas as pd
from tabulate import tabulate

project_path = os.getcwd()
db_directory = os.getenv('DB_DIR', f"{project_path}/data/db/")
input_directory = os.getenv('INPUT_DIR', f"{project_path}/data/input/")
output_directory = os.getenv('OUTPUT_DIR', f"{project_path}/data/output/")

###
# LOAD MODEL
###

model = Model(hidden_channels=32).to(device)
# model.load_state_dict(torch.load(f"{db_directory}model12125.pt", weights_only=True))
model.load_state_dict(torch.load(f"{db_directory}model12125.pt", map_location=torch.device('cpu')))
model.eval()

###
# GET USER INPUT
###

# get maps of disease and drug to their node ids
disease_id_table = pd.read_csv(f"{db_directory}disease_id.csv")
drug_id_table = pd.read_csv(f"{db_directory}drug_id.csv")

# TODO - get user input with docker
# output table of disease options
print(tabulate(disease_id_table, headers=["ID","Disease"], tablefmt='psql', showindex="never"))
disease_id = int(input("\nENTER THE ID OF A DISEASE IN THE TABLE: "))
disease_string = disease_id_table.iloc[disease_id, disease_id_table.columns.get_loc('name')]

# input(f"CHOSEN DISEASE (Enter/Return): {disease_string}")

# temporarily read input from file
# f = open(f"{input_directory}disease.txt", "r")
# disease_id = int(f.read())
# disease_string = disease_id_table.iloc[disease_id, disease_id_table.columns.get_loc('name')]

###
# GET PREDICTIONS
###

drug_indices = data['Drug'].node_id.tolist()
disease_indices = data['Disease'].node_id.tolist()

# get the prediction score for the edge between the chosen disease and all possible drugs
curr_disease_edges = torch.tensor([drug_indices, [disease_id] * len(drug_indices)], dtype=torch.int64)
pred = model(data.x_dict, data.edge_index_dict, curr_disease_edges)

drug_list = drug_id_table['name'].tolist()

# compute drug, id, and prediction score for each edge
predicted_new_edges = [drug_list, [disease_string] * len(drug_indices), pred.tolist()]
# convert predictions into a dataframe format
transposed_edges = [list(row) for row in zip(*predicted_new_edges)]

# store only the top 250 predicted edges for each disease
transposed_edges_sorted = sorted(transposed_edges, key=lambda x: x[2], reverse=False)[:250]

###
# SAVE PREDICTIONS
###

# get today's date
today = datetime.date.today()

# remove non digits or characters from the disease name
stripped_disease_string = re.sub(r'[^a-zA-Z0-9]', '', disease_string)

# save top 250 predictions to a csv file
with open(f"{output_directory}{today}_{disease_string.replace(" ","")}_candidateDrugs.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["drug", "disease", "prediction_score"])
    writer.writerows(transposed_edges_sorted)

# print(f"\nCANDIDATE DRUGS SAVED TO {output_directory}{today}_{stripped_disease_string}_candidateDrugs.csv")