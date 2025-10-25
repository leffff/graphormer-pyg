import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import Dataset
from tqdm import tqdm
from torch_geometric.nn.pool import global_mean_pool

from torch import nn
from graphormer.model import Graphormer
from graphormer.functional import precalculate_custom_attributes, precalculate_paths

import time

# Install rdkit
# !pip install rdkit

# Load dataset
dataset = MoleculeNet(root="./", name="ESOL")
print(dataset)

# HYPER-PARAMETERS
NUM_LAYERS = 3
NODE_DIM = 128
FF_DIM = 256
N_HEADS = 4
MAX_IN_DEGREE = 5
MAX_OUT_DEGREE = 5
MAX_PATH_DISTANCE = 5

# Create model
model = Graphormer(
    num_layers=NUM_LAYERS,
    input_node_dim=dataset.num_node_features,
    node_dim=NODE_DIM,
    input_edge_dim=dataset.num_edge_features,
    edge_dim=NODE_DIM,
    output_dim=dataset[0].y.shape[1],
    n_heads=N_HEADS,
    ff_dim=FF_DIM,
    max_in_degree=MAX_IN_DEGREE,
    max_out_degree=MAX_OUT_DEGREE,
    max_path_distance=MAX_PATH_DISTANCE,
)

# precalculate attributes for each graph
modified_data_list = []
for data in dataset:
    modified_data = precalculate_custom_attributes(data, max_in_degree=MAX_IN_DEGREE, max_out_degree=MAX_OUT_DEGREE)
    modified_data_list.append(modified_data)

class ModifiedDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list        
    def __len__(self):
        return len(self.data_list)    
    def __getitem__(self, idx):
        return self.data_list[idx]

modified_dataset = ModifiedDataset(modified_data_list)

# Dataset splitting
from sklearn.model_selection import train_test_split
test_ids, train_ids = train_test_split([i for i in range(len(modified_dataset))], test_size=0.8, random_state=42)
train_loader = DataLoader(Subset(modified_dataset, train_ids), batch_size=8)
test_loader = DataLoader(Subset(modified_dataset, test_ids), batch_size=8)

# precalculate node_paths_length, edge_paths_tensor and edge_paths_length for each batch
train_node_edge_paths = []
for batch in train_loader:
    _, _, node_paths_length, edge_paths_tensor, edge_paths_length = precalculate_paths(batch, max_path_distance=MAX_PATH_DISTANCE)
    train_node_edge_paths.append((node_paths_length, edge_paths_tensor, edge_paths_length))
test_node_edge_paths = []
for batch in test_loader:
    _, _, node_paths_length, edge_paths_tensor, edge_paths_length = precalculate_paths(batch, max_path_distance=MAX_PATH_DISTANCE)
    test_node_edge_paths.append((node_paths_length, edge_paths_tensor, edge_paths_length))

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_function = nn.L1Loss(reduction="sum") 

# Training and evaluation
DEVICE = "cuda"

# Ensure CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

model.to(DEVICE)
for epoch in range(10):
    model.train()
    batch_loss = 0.0
    epoch_start = time.time()
    
    for i, batch in enumerate(tqdm(train_loader)):
        node_paths_length, edge_paths_tensor, edge_paths_length = train_node_edge_paths[i]
        batch.node_paths_length = node_paths_length
        batch.edge_paths_tensor = edge_paths_tensor
        batch.edge_paths_length = edge_paths_length

        batch = batch.to(DEVICE) 
        y = batch.y
        optimizer.zero_grad()

        start_forward = time.time()
        output = global_mean_pool(model(batch), batch.batch)
        loss = loss_function(output, y)
        batch_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1} - TRAIN_LOSS: {batch_loss / len(train_ids):.6f}, Time: {epoch_time:.2f}s")

    model.eval()
    batch_loss = 0.0
    for i, batch in enumerate(tqdm(test_loader)):
        node_paths, edge_paths, node_paths_length, edge_paths_tensor, edge_paths_length = test_node_edge_paths[i]
        batch.node_paths = node_paths
        batch.edge_paths = edge_paths
        batch.node_paths_length = node_paths_length
        batch.edge_paths_tensor = edge_paths_tensor
        batch.edge_paths_length = edge_paths_length
        batch = batch.to(DEVICE) 
        y = batch.y
        with torch.no_grad():
            output = global_mean_pool(model(batch), batch.batch)
            loss = loss_function(output, y)
            
        batch_loss += loss.item()

    print("EVAL LOSS", batch_loss / len(test_ids))