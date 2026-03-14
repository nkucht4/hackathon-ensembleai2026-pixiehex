import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

df = pd.read_parquet('chebi_dataset_train.parquet')
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def preprocess_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    fp = gen.GetFingerprint(mol)
    fp_array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    return fp_array
def preprocessing():

    print("Przetwarzanie cząsteczek...")
    df['fingerprint'] = df['SMILES'].apply(preprocess_smiles)

    df = df.dropna(subset=['fingerprint'])

    # X = np.stack(df['fingerprint'].values)
    # y = df['target_column'].values

    print(f"Kształt: {df.shape}")
    print(df['fingerprint'].head())

    return df




