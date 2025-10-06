import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, calc_dihedral, Vector
import torch_cluster
import torch_geometric

# CONSTANTS
AA_LETTER_TO_INDEX = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
    'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
    'X': 20, '-': 21
}
INDEX_TO_AA = {v: k for k, v in AA_LETTER_TO_INDEX.items()}

# Physicochemical properties
HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0
}

CHARGE = {
    'R': 1.0, 'K': 1.0, 'H': 0.1,  # positive
    'D': -1.0, 'E': -1.0,  # negative
    'X': 0.0
}

VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0,
    'X': 100.0
}

# Secondary structure propensities (Chou-Fasman)
HELIX_PROPENSITY = {
    'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'K': 1.16,
    'F': 1.13, 'Q': 1.11, 'W': 1.08, 'I': 1.08, 'V': 1.06,
    'D': 1.01, 'H': 1.00, 'R': 0.98, 'T': 0.83, 'S': 0.77,
    'C': 0.70, 'Y': 0.69, 'N': 0.67, 'P': 0.57, 'G': 0.57,
    'X': 1.0
}

BETA_PROPENSITY = {
    'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37,
    'L': 1.30, 'T': 1.19, 'C': 1.19, 'Q': 1.10, 'M': 1.05,
    'R': 0.93, 'N': 0.89, 'H': 0.87, 'A': 0.83, 'G': 0.75,
    'S': 0.75, 'K': 0.74, 'P': 0.55, 'D': 0.54, 'E': 0.37,
    'X': 1.0
}

TURN_PROPENSITY = {
    'G': 1.56, 'N': 1.56, 'P': 1.52, 'D': 1.46, 'S': 1.43,
    'C': 1.19, 'Y': 1.14, 'K': 1.01, 'Q': 0.98, 'T': 0.96,
    'W': 0.81, 'R': 0.79, 'H': 0.75, 'E': 0.74, 'A': 0.66,
    'M': 0.60, 'F': 0.60, 'L': 0.59, 'V': 0.50, 'I': 0.47,
    'X': 1.0
}

# Special residue sets
AROMATIC = {'F', 'Y', 'W', 'H'}
HYDROPHOBIC = {'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W'}
CHARGED = {'R', 'K', 'D', 'E', 'H'}
POSITIVE = {'R', 'K', 'H'}
NEGATIVE = {'D', 'E'}

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

def compute_physicochemical_features(seq):
    """Compute physicochemical properties per residue"""
    features = []
    for aa in seq:
        hydrophobicity = HYDROPHOBICITY.get(aa, 0.0) / 5.0  # normalize to ~[-1, 1]
        charge = CHARGE.get(aa, 0.0)
        volume = VOLUME.get(aa, 100.0) / 200.0  # normalize
        features.append([hydrophobicity, charge, volume])
    return torch.tensor(features, dtype=torch.float32)

def compute_ss_propensities(seq):
    """Compute secondary structure propensities"""
    features = []
    for aa in seq:
        helix = HELIX_PROPENSITY.get(aa, 1.0)
        beta = BETA_PROPENSITY.get(aa, 1.0)
        turn = TURN_PROPENSITY.get(aa, 1.0)
        features.append([helix, beta, turn])
    return torch.tensor(features, dtype=torch.float32)

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

def compute_special_residue_flags(seq):
    """Binary flags for special residues"""
    features = []
    for aa in seq:
        is_proline = float(aa == 'P')
        is_glycine = float(aa == 'G')
        is_aromatic = float(aa in AROMATIC)
        is_cysteine = float(aa == 'C')
        features.append([is_proline, is_glycine, is_aromatic, is_cysteine])
    return torch.tensor(features, dtype=torch.float32)

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

def compute_interaction_features(edge_index, seq, coords):
    """Compute interaction-based edge features"""
    i, j = edge_index[0], edge_index[1]
    n_edges = edge_index.shape[1]
    
    # Get atom coordinates
    N = coords[:, 0]
    O = coords[:, 3] if coords.shape[1] > 3 else coords[:, 2]  # O or C if O missing
    
    # Potential H-bonds (N...O distance)
    is_potential_hbond = torch.zeros(n_edges)
    for idx in range(n_edges):
        if abs(i[idx] - j[idx]) >= 3:  # not too close in sequence
            dist_NO = torch.norm(N[i[idx]] - O[j[idx]])
            dist_ON = torch.norm(O[i[idx]] - N[j[idx]])
            if dist_NO < 3.5 or dist_ON < 3.5:
                is_potential_hbond[idx] = 1.0
    
    # Salt bridges (opposite charges)
    is_potential_salt_bridge = torch.zeros(n_edges)
    for idx in range(n_edges):
        aa_i, aa_j = seq[i[idx]], seq[j[idx]]
        if (aa_i in POSITIVE and aa_j in NEGATIVE) or (aa_i in NEGATIVE and aa_j in POSITIVE):
            ca_dist = torch.norm(coords[i[idx], 1] - coords[j[idx], 1])
            if ca_dist < 6.0:  # reasonable salt bridge distance
                is_potential_salt_bridge[idx] = 1.0
    
    # Aromatic-aromatic interactions
    is_aromatic_aromatic = torch.zeros(n_edges)
    for idx in range(n_edges):
        if seq[i[idx]] in AROMATIC and seq[j[idx]] in AROMATIC:
            ca_dist = torch.norm(coords[i[idx], 1] - coords[j[idx], 1])
            if ca_dist < 7.0:
                is_aromatic_aromatic[idx] = 1.0
    
    # Hydrophobic contacts
    is_hydrophobic_contact = torch.zeros(n_edges)
    for idx in range(n_edges):
        if seq[i[idx]] in HYDROPHOBIC and seq[j[idx]] in HYDROPHOBIC:
            ca_dist = torch.norm(coords[i[idx], 1] - coords[j[idx], 1])
            if ca_dist < 8.0:
                is_hydrophobic_contact[idx] = 1.0
    
    # Disulfide candidates
    is_disulfide_candidate = torch.zeros(n_edges)
    for idx in range(n_edges):
        if seq[i[idx]] == 'C' and seq[j[idx]] == 'C':
            ca_dist = torch.norm(coords[i[idx], 1] - coords[j[idx], 1])
            if 3.5 < ca_dist < 6.5:  # typical disulfide CA-CA distance
                is_disulfide_candidate[idx] = 1.0
    
    return torch.stack([
        is_potential_hbond,
        is_potential_salt_bridge,
        is_aromatic_aromatic,
        is_hydrophobic_contact,
        is_disulfide_candidate
    ], dim=1)

def parse_peptide_pdb(pdb_path, chain_id=None):
    """Parse PDB file with support for modified residues"""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure('peptide', pdb_path)
    model = struct[0]
    chain = model[chain_id] if chain_id else next(model.get_chains())
    atoms, coords, seq, mask = ['N', 'CA', 'C', 'O'], [], [], []
    
    from Bio.PDB.Polypeptide import three_to_one
    
    for res in chain:
        if res.id[0] != ' ':
            # Check if it's a modified amino acid in peptide chain
            if not hasattr(res, 'resname') or res.resname not in ['PTR', 'SEP', 'TPO']:
                continue
        
        # Handle modified residues
        resname = res.resname
        if resname == 'PTR':  # Phosphotyrosine -> Y
            aa = 'Y'
        elif resname == 'SEP':  # Phosphoserine -> S
            aa = 'S'
        elif resname == 'TPO':  # Phosphothreonine -> T
            aa = 'T'
        elif resname == 'DIY':  # Some other modification, map to closest
            aa = 'Y'
        else:
            try:
                aa = three_to_one(resname)
            except KeyError:
                aa = 'X'
        
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
    
    # Physicochemical features
    physicochemical = compute_physicochemical_features(seq)
    
    # Secondary structure propensities
    ss_propensities = compute_ss_propensities(seq)
    
    # Structural features
    backbone_curve = compute_backbone_curvature(CA)
    n_contacts, local_density = compute_local_features(CA)
    
    # Special residue flags
    special_flags = compute_special_residue_flags(seq)
    
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
    
    # Interaction features
    interaction_features = compute_interaction_features(edge_index, seq, coords)
    
    # Combine all edge scalar features
    edge_s = torch.cat([
        rbf(edge_vecs.norm(dim=-1), D_count=num_rbf),  # [E, 16]
        get_posenc(edge_index, num_posenc),            # [E, 16]
        edge_types,                                    # [E, 4]
    ], dim=-1)  # Total: [E,36]
    
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
                f.write(f'  node_s: {tuple(graph["node_s"].shape)}\n')
                f.write(f'  node_v: {tuple(graph["node_v"].shape)}\n')
                f.write(f'  edge_s: {tuple(graph["edge_s"].shape)}\n')
                f.write(f'  edge_v: {tuple(graph["edge_v"].shape)}\n')
                f.write(f'  edge_index: {tuple(graph["edge_index"].shape)}\n')
                f.write('\n')
                
            except Exception as e:
                print(f"\nError processing {pdb_path}: {e}")
                continue

    print(f"\n✅ Done! Saved enhanced .pt files to {OUTPUT_DIR}")
    print(f"Node features: 10 dimensions (3 dihedrals + 7 peptide-specific)")
    print(f"Edge features: 36 dimensions (16 RBF + 16 posenc + 4 peptide-specific)")
