import boto3
import torch, csv, os, datetime, re
from model import Model, device, data
import pandas as pd
import logging
import warnings
import json
import base64

logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)

run_env = os.getenv('RUN_ENV', 'local')
if run_env == 'fargate':
    logger.info("RUNNING PREDICT.PY IN FARGATE ENVIRONMENT")
    session = boto3.Session()
    s3_client = session.client('s3')
    bucket_name = os.getenv('S3_BUCKET_NAME', 'kearneyneurokb-dr')

    db_directory = '/tmp/db/'
    output_directory = '/tmp/output/'

    session_id = os.getenv('SESSION_ID')
    prefix = f"{session_id}/" if session_id else ""
    logger.info(f"session: {session_id}")
    parameters = os.getenv('PARAMETERS')
    parameters = json.loads(base64.b64decode(parameters).decode('utf-8')) if parameters else None
    logger.info(f"parameters: {parameters}")

    os.makedirs(output_directory, exist_ok=True)
else:
    logger.info(f"RUNNING IN {run_env} ENVIRONMENT")
    base_directory = os.path.dirname(os.path.dirname(__file__))
    db_directory = os.getenv('DB_DIR', os.path.join(base_directory, 'data/db/'))
    input_directory = os.getenv('INPUT_DIR', os.path.join(base_directory, 'data/input/'))
    output_directory = os.getenv('OUTPUT_DIR', os.path.join(base_directory, 'data/output/'))

disease_env = os.getenv('DISEASE', 'disease')
logger.info("LOADED S3 FILES")

###
# LOAD MODEL
###

model = Model(hidden_channels=32).to(device)
model.load_state_dict(torch.load(f"{db_directory}model12125.pt", map_location=torch.device('cpu')))
model.eval()
logger.info("LOADED MODEL")

###
# GET USER INPUT
###

# get maps of disease and drug to their node ids
disease_id_table = pd.read_csv(f"{db_directory}disease_id.csv")
drug_id_table = pd.read_csv(f"{db_directory}drug_id.csv")
logger.info("READ CSV FILES")
# TODO - get user input with docker
# output table of disease options
# print(tabulate(disease_id_table, headers=["ID","Disease"], tablefmt='psql', showindex="never"))
# disease_id = int(input("\nENTER THE ID OF A DISEASE IN THE TABLE: "))
# disease_string = disease_id_table.iloc[disease_id, disease_id_table.columns.get_loc('name')]
# input(f"CHOSEN DISEASE (Enter/Return): {disease_string}")

# temporarily read input from file
# f = open(f"{input_directory}disease.txt", "r")

disease_id = int(disease_env)
disease_string = disease_id_table.iloc[disease_id, disease_id_table.columns.get_loc('name')]

###
# GET PREDICTIONS
###

drug_indices = data['Drug'].node_id.tolist()
drug_indices = drug_indices
disease_indices = data['Disease'].node_id.tolist()

# get the prediction score for the edge between the chosen disease and all possible drugs
curr_disease_edges = torch.tensor([drug_indices, [disease_id] * len(drug_indices)], dtype=torch.int64)
logger.info("MADE EDGES")
logger.info(f"LEN EDGES: {len(curr_disease_edges)}")
pred = model(data.x_dict, data.edge_index_dict, curr_disease_edges)

logger.info("GOT PREDICTIONS")
drug_list = drug_id_table['name'].tolist()

# compute drug, id, and prediction score for each edge
predicted_new_edges = [drug_list, [disease_string] * len(drug_indices), pred.tolist()]
# convert predictions into a dataframe format
transposed_edges = [list(row) for row in zip(*predicted_new_edges)]

# store only the top 25 predicted edges for each disease
transposed_edges_sorted = sorted(transposed_edges, key=lambda x: x[2], reverse=False)[:25]
logger.info("FORMATTED PREDICTIONS")
###
# SAVE PREDICTIONS
###

# get today's date
today = datetime.date.today()

# remove non digits or characters from the disease name
stripped_disease_string = re.sub(r'[^a-zA-Z0-9]', '', disease_string)

output_file = f"{output_directory}{today}_{disease_string.replace(' ', '')}_candidateDrugs.csv"

# save top 250 predictions to a csv file
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["drug", "disease", "prediction_score"])
    writer.writerows(transposed_edges_sorted)

if run_env == 'fargate':
    s3_client.upload_file(output_file, bucket_name, f"{prefix}output/{today}_{disease_string.replace(' ', '')}_candidateDrugs.csv")
    logger.info("WROTE TO S3 FILE")
else:
    logger.info("WROTE TO LOCAL FILE")
    print(f"\nCANDIDATE DRUGS SAVED TO {output_file}")