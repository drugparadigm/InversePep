import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, calc_dihedral, Vector
import torch_cluster
import torch_geometric
from Bio.Data.IUPACData import protein_letters_3to1

# CONSTANTS
AA_LETTER_TO_INDEX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, '-': 21
}
INDEX_TO_AA = {v: k for k, v in AA_LETTER_TO_INDEX.items()}

# UTILITY FUNCTIONS
def normalize(tensor, dim=-1):
    return torch.nan_to_num(
        tensor / torch.linalg.norm(tensor, dim=dim, keepdim=True)
    )

def rbf(D, D_min=0., D_max=20., D_count=16):
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device).view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    return torch.exp(-((D.unsqueeze(-1) - D_mu) / D_sigma) ** 2)

def get_posenc(edge_index, num_posenc=16):
    d = edge_index[0] - edge_index[1]
    freq = torch.exp(
        torch.arange(0, num_posenc, 2, device=d.device, dtype=torch.float32)
        * -(np.log(10000.0) / num_posenc)
    )
    angles = d.unsqueeze(-1) * freq
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

def get_orientations_single(CA_coords):
    forward = normalize(CA_coords[1:] - CA_coords[:-1])
    backward = normalize(CA_coords[:-1] - CA_coords[1:])
    forward = F.pad(forward, (0, 0, 0, 1))
    backward = F.pad(backward, (0, 0, 1, 0))
    return torch.cat([forward.unsqueeze(1), backward.unsqueeze(1)], dim=1)

def compute_backbone_dihedrals(coords):
    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    Np = torch.roll(N, shifts=-1, dims=0)
    CA_prev = torch.roll(CA, shifts=1, dims=0)
    C_prev = torch.roll(C, shifts=1, dims=0)
    phi, psi, omega = [], [], []
    for i in range(len(coords)):
        if 0 < i < len(coords) - 1:
            v_phi = calc_dihedral(Vector(C_prev[i]), Vector(N[i]), Vector(CA[i]), Vector(C[i]))
            v_psi = calc_dihedral(Vector(N[i]), Vector(CA[i]), Vector(C[i]), Vector(Np[i]))
            v_omega = calc_dihedral(Vector(CA[i]), Vector(C[i]), Vector(Np[i]), Vector(CA[i+1]))
        else:
            v_phi = v_psi = v_omega = 0.0
        phi.append(v_phi)
        psi.append(v_psi)
        omega.append(v_omega)
    return torch.tensor([phi, psi, omega], dtype=torch.float32).T

# NODE FEATURE FUNCTIONS
def compute_terminal_features(seq_len):
    """Compute terminal-related features"""
    n_term_mask = torch.zeros(seq_len)
    c_term_mask = torch.zeros(seq_len)
    n_term_mask[0] = 1.0
    c_term_mask[-1] = 1.0
    
    # Distance from terminals (normalized)
    dist_from_n = torch.arange(seq_len, dtype=torch.float32) / max(seq_len - 1, 1)
    dist_from_c = torch.flip(dist_from_n, [0])
    
    return torch.stack([n_term_mask, c_term_mask, dist_from_n, dist_from_c], dim=1)



def compute_backbone_curvature(ca_coords):
    """Compute local backbone curvature"""
    curvature = []
    for i in range(len(ca_coords)):
        if i == 0 or i == len(ca_coords) - 1:
            curvature.append(0.0)
        else:
            v1 = normalize(ca_coords[i] - ca_coords[i-1])
            v2 = normalize(ca_coords[i+1] - ca_coords[i])
            cos_angle = torch.clamp(torch.dot(v1, v2), -1.0, 1.0)
            angle = torch.acos(cos_angle).item()
            curvature.append(angle / np.pi)  # normalize to [0, 1]
    return torch.tensor(curvature, dtype=torch.float32).unsqueeze(1)

def compute_local_features(ca_coords, contact_threshold=8.0, density_radius=10.0):
    """Compute local density and contact count"""
    n_residues = len(ca_coords)
    dist_matrix = torch.cdist(ca_coords, ca_coords)
    
    # Contact count (within threshold, excluding self)
    contacts = ((dist_matrix < contact_threshold) & (dist_matrix > 0)).float()
    n_contacts = contacts.sum(dim=1) / n_residues  # normalize
    
    # Local density (within radius)
    density = ((dist_matrix < density_radius) & (dist_matrix > 0)).float()
    local_density = density.sum(dim=1) / n_residues  # normalize
    
    return n_contacts.unsqueeze(1), local_density.unsqueeze(1)



# EDGE FEATURE FUNCTIONS
def compute_edge_types(edge_index, seq_len):
    """Compute edge type indicators"""
    i, j = edge_index[0], edge_index[1]
    seq_dist = torch.abs(i - j)
    
    is_sequential = (seq_dist == 1).float()
    is_i_plus_2 = (seq_dist == 2).float()
    is_i_plus_3 = (seq_dist == 3).float()
    is_i_plus_4 = (seq_dist == 4).float()
    
    return torch.stack([is_sequential, is_i_plus_2, is_i_plus_3, is_i_plus_4], dim=1)


def parse_peptide_pdb(pdb_path, chain_id=None):
    """Parse PDB file with support for modified residues"""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('peptide', pdb_path)
    model = struct[0]
    chain = model[chain_id] if chain_id else next(model.get_chains())
    atoms, coords, seq, mask = ['N', 'CA', 'C', 'O'], [], [], []

    for res in chain:
        if res.id[0] != ' ':
            # Check if it's a modified amino acid in peptide chain
            if not hasattr(res, 'resname') or res.resname not in ['PTR', 'SEP', 'TPO']:
                continue

        # Handle modified residues
        resname = res.resname.strip()
        if resname == 'PTR':  # Phosphotyrosine -> Y
            aa = 'Y'
        elif resname == 'SEP':  # Phosphoserine -> S
            aa = 'S'
        elif resname == 'TPO':  # Phosphothreonine -> T
            aa = 'T'
        elif resname == 'DIY':
            aa = 'Y'
        else:
            aa = protein_letters_3to1.get(resname.capitalize(), 'X')

        seq.append(aa)
        atom_coords, present = [], True
        for atom in atoms:
            if atom in res:
                atom_coords.append(res[atom].get_coord())
            else:
                atom_coords.append(np.zeros(3))
                present = False
        coords.append(atom_coords)
        mask.append(present)

    return ''.join(seq), np.array(coords, dtype=np.float32), np.array(mask, dtype=bool)


@torch.no_grad()
def construct_peptide_data_enhanced(coords, seq, mask,
                                   num_posenc=16, num_rbf=16, knn_num=10):
    """Construct enhanced peptide data with all features"""
    coords = torch.as_tensor(coords)
    seq_idx = torch.tensor([AA_LETTER_TO_INDEX.get(a, AA_LETTER_TO_INDEX['X'])
                             for a in seq], dtype=torch.long)
    
    dihedrals = compute_backbone_dihedrals(coords)
    CA = coords[:, 1]
    edge_index = torch_cluster.knn_graph(CA, k=knn_num)
    edge_index = torch_geometric.utils.coalesce(edge_index)
    
    # NODE FEATURES
    # Terminal features
    terminal_features = compute_terminal_features(len(seq))
    
    # Structural features
    backbone_curve = compute_backbone_curvature(CA)
    n_contacts, local_density = compute_local_features(CA)
    
    # Combine all node scalar features
    node_s = torch.cat([
        dihedrals,           # [N, 3]
        terminal_features,   # [N, 4]
        backbone_curve,      # [N, 1]
        local_density,       # [N, 1]
        n_contacts,          # [N, 1]
    ], dim=-1)  # Total: [N, 10]
    
    # Node vector features
    node_v = get_orientations_single(CA)
    
    # EDGE FEATURES
    edge_vecs = CA[edge_index[0]] - CA[edge_index[1]]
    
    # Edge type features
    edge_types = compute_edge_types(edge_index, len(seq))
    
    # Combine all edge scalar features
    edge_s = torch.cat([
        rbf(edge_vecs.norm(dim=-1), D_count=num_rbf),  # [E, 16]
        get_posenc(edge_index, num_posenc),            # [E, 16]
        edge_types,                                    # [E, 4]
    ], dim=-1)  # Total: [E, 36]
    
    # Edge vector features
    edge_v = normalize(edge_vecs).unsqueeze(1)
    
    return {
        'seq': seq_idx,
        'coords': coords,
        'node_s': node_s,
        'node_v': node_v,
        'edge_s': edge_s,
        'edge_v': edge_v,
        'edge_index': edge_index,
        'mask': torch.tensor(mask),
        # Additional Meta Data
        'seq_len': len(seq),
        'seq_str': seq
    }

def peptide2data_enhanced(pdb_path, num_posenc=16, num_rbf=16, knn_num=10):
    """Enhanced wrapper for peptide data processing"""
    seq, coords, mask = parse_peptide_pdb(pdb_path)
    return construct_peptide_data_enhanced(coords, seq, mask,
                                          num_posenc=num_posenc,
                                          num_rbf=num_rbf,
                                          knn_num=knn_num)

if __name__ == '__main__':
    csv_file = 'final_merged_dataset.csv'
    OUTPUT_DIR = 'peptides_pt_enhanced_latest_3'
    shapes_file = 'PPshapes.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.read_csv(csv_file)
    pdb_paths = df['pdb_path'].tolist()
    
    print(f"Processing {len(pdb_paths)} PDB files with enhanced features...")
    
    with open(shapes_file, 'w') as f:
        for pdb_path in tqdm(pdb_paths, desc='Preprocessing PDBs'):
            try:
                graph = peptide2data_enhanced(pdb_path)
                name = os.path.splitext(os.path.basename(pdb_path))[0]
                
                torch.save(graph, os.path.join(OUTPUT_DIR, f'{name}.pt'))
                
                f.write(f'{name}:\n')
                f.write(f'  seq_len: {graph["seq_len"]}\n')
                f.write(f'  seq_str: {graph["seq_str"]}\n')
                f.write(f'  node_s: {tuple(graph["node_s"].shape)} (10 features)\n')
                f.write(f'  node_v: {tuple(graph["node_v"].shape)}\n')
                f.write(f'  edge_s: {tuple(graph["edge_s"].shape)} (36 features)\n')
                f.write(f'  edge_v: {tuple(graph["edge_v"].shape)}\n')
                f.write(f'  edge_index: {tuple(graph["edge_index"].shape)}\n')
                f.write('\n')
                
            except Exception as e:
                print(f"\nError processing {pdb_path}: {e}")
                continue

    print(f"\n✅ Done! Saved enhanced .pt files to {OUTPUT_DIR}")
    
