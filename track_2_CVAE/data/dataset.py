"""
Track 2 CVAE — PyTorch Dataset
"""
import torch
from torch.utils.data import Dataset
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import encode_voyage, build_condition_vector


class VoyageDataset(Dataset):
    def __init__(self, df, artefacts):
        self.df = df.reset_index(drop=True)
        self.artefacts = artefacts

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        a   = self.artefacts

        port_indices, time_feats, mask = encode_voyage(
            row,
            a['port2idx'], a['PAD_IDX'], a['END_IDX'], a['UNK_IDX'],
            a['MAX_TRANSIT'], a['MAX_DWELL'],
        )
        c_vec = torch.tensor(
            build_condition_vector(row, a['vessel_type_map'], a['nz_port_map']),
            dtype=torch.float32,
        )

        # targets = same as inputs (reconstruction)
        port_target    = port_indices.clone()
        transit_target = time_feats[:, 0].clone()
        dwell_target   = time_feats[:, 1].clone()

        return (port_indices, time_feats, c_vec,
                port_target, transit_target, dwell_target, mask)
