import pandas as pd
from preprocessing import preprocessing
from model import MoleculeGCN
import torch
# df = pd.read_parquet('chebi_submission_example.parquet')

# print(df.head())

df = pd.read_parquet('chebi_dataset_train.parquet')

loader, data_list = preprocessing(df)
input_dim = data_list[0].num_node_features
model = MoleculeGCN(hidden_channels=64, input_dim=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


print("Rozpoczynanie treningu...")
model.train()
for epoch in range(1, 101):
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    if epoch % 10 == 0:
        print(f'Epoka: {epoch:03d}, Loss: {total_loss / len(loader.dataset):.4f}')

print("Trening zakończony!")
    
    # E. Zapis modelu
torch.save(model.state_dict(), 'molecule_gnn_model.pth')
print("Model zapisany jako molecule_gnn_model.pth")

