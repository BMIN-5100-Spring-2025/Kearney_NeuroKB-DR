"""
This file runs a simple link prediction model using SAGEConv to predict the
DRUGINDICATIONDISEASE edge between Drug and Disease nodes. Loss and RMSE
are reported at the end of the epochs. Loss and true labels are saved into
CSVs.
"""
###
# IMPORTS
###

import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric
import pandas as pd
from sklearn.metrics import roc_auc_score
import time
from app.model import Model

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

model = Model(hidden_channels=32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

###
# DEFINE TRAINING AND TESTING PROCESSES
###

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

###
# TRAIN MODEL
###

if __name__ == '__main__':
    start_time = time.time()

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

    ###
    # SAVE LOSS AND LABELS TO CSV
    ###

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