import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit import Chem

# --- 1. FUNKCJE POMOCNICZE DO CECH ATOMÓW ---

def get_atom_features(atom):
    """Tworzy wektor cech dla pojedynczego atomu (One-Hot Encoding)."""
    # Lista najczęstszych atomów w chemii medycznej
    symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    symbol = atom.GetSymbol()
    
    # One-hot encoding symbolu
    v = [1 if symbol == s else 0 for s in symbols]
    # Dodatkowe cechy: stopień, ładunek, hybrydyzacja
    v.append(atom.GetDegree())
    v.append(atom.GetFormalCharge())
    v.append(int(atom.GetIsAromatic()))
    return v

def smiles_to_graph(smiles, target):
    """Zamienia SMILES na obiekt grafu PyTorch Geometric."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None

    node_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    y = torch.tensor([[target]], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

def preprocessing(df):
    data_list = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row['smiles'], row['target'])
        if g:
            data_list.append(g)
    loader = DataLoader(data_list, batch_size=32, shuffle=True)

    return loader

    


