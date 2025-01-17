"""
This file runs a simple link prediction model using SAGEConv to predict the
DRUGINDICATIONDISEASE edge between Drug and Disease nodes. Loss and RMSE
is visualized at the end of the epochs.
"""
###
# IMPORTS
###

from torch_geometric.nn import SAGEConv,to_hetero
import os, sys, csv
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.nn import Linear
import torch_geometric
import pandas as pd
from sklearn.metrics import roc_auc_score
import time

project_path = os.path.dirname(os.getcwd())

###
# LOAD & SPLIT DATA
###

device = torch_geometric.device('auto')

data = torch.load(f'{project_path}/data/neuroKB.pth')
data = T.ToUndirected()(data).to(device)

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
    is_undirected=True,
    edge_types = [("Drug", "DRUGINDICATIONDISEASE", "Disease")]
)(data)

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
    def forward(self, x, edge_index):
        # TODO - could adjust relu to something else
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['Drug'][row], z_dict['Disease'][col]], dim=-1)
        z = self.lin1(z)
        z = torch.sigmoid(z)
        z = self.lin2(z)
        return z.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)

model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()
    optimizer.zero_grad()
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data['Drug','DRUGINDICATIONDISEASE','Disease'].edge_label_index)
    target = train_data['Drug','DRUGINDICATIONDISEASE','Disease'].edge_label
    loss = F.binary_cross_entropy_with_logits(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['Drug','DRUGINDICATIONDISEASE','Disease'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['Drug','DRUGINDICATIONDISEASE','Disease'].edge_label.float()
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)
    return [float(bce_loss), pred, target]

if __name__ == '__main__':
    start_time = time.time()
    ###
    # RUN MODEL
    ###

    epochs = range(1, 200)
    losses = []
    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in epochs:
        loss = train()
        train_loss = test(train_data)[0]
        val_loss = test(val_data)[0]
        test_loss = test(test_data)

        losses.append(loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss[0])

        pred = test_loss[1]
        target = test_loss[2]

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_loss:.4f}, '
              f'Val: {val_loss:.4f}, Test: {test_loss[0]:.4f}')

    total_loss= pd.DataFrame(list(zip(losses, train_losses, val_losses, test_losses)),
                        columns=["Loss", "Train", "Val", "Test"])
    total_loss.to_csv(f'{project_path}/data/losses.csv', index=False)

    pred_target = pd.DataFrame(list(zip(pred.tolist(), target.tolist())),
                               columns=["pred", "target"])
    pred_target.to_csv(f'{project_path}/data/preds.csv', index=False)

    auroc = roc_auc_score(target.tolist(), pred.tolist())
    print(f"\nAUROC: {auroc}")

    elapsed_time = round(time.time() - start_time, 3)/60
    print(f"\n--- {elapsed_time} mins ---")

    ###
    # APPLY MODEL TO PREDICT NEW EDGES BY DISEASE
    ###

    drug_indices = data['Drug'].node_id.tolist()
    disease_indices = data['Disease'].node_id.tolist()

    all_top_preds = []

    # iterate through each disease and check that disease to all the drugs and get that prediction score
    for i in disease_indices:
        curr_disease_edges = torch.tensor([drug_indices, [i] * len(drug_indices)], dtype=torch.int64)

        pred = model(data.x_dict, data.edge_index_dict, curr_disease_edges)

        predicted_new_edges = [drug_indices, [i] * len(drug_indices), pred.tolist()]
        transposed_edges = [list(row) for row in zip(*predicted_new_edges)]
        # store only the top 250 predicted edges for each disease
        transposed_edges_sorted = sorted(transposed_edges, key=lambda x: x[2], reverse=True)[:250]
        all_top_preds += transposed_edges_sorted

    # save all the best score predictions for all diseases
    with open(f'{project_path}/data/top_predictions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(all_top_preds)