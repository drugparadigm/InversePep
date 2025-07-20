import torch

def normalize(tensor, dim=-1, eps=1e-8):
    """
    Normalize a tensor along a dimension with NaN protection.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor with unit norm along specified dimension
    """
    return nan_to_num(torch.div(tensor, norm(tensor, dim=dim, keepdim=True, eps=eps)))

def nan_to_num(ts, val=0.0):
    """
    Replace NaN, Inf, and -Inf values with a specified value.
    
    Args:
        ts: Input tensor
        val: Value to replace non-finite elements with
        
    Returns:
        Tensor with non-finite values replaced
    """
    val = torch.tensor(val, dtype=ts.dtype, device=ts.device)
    return torch.where(~torch.isfinite(ts), val, ts)

def norm(tensor, dim, eps=1e-8, keepdim=False):
    """
    Compute the L2 norm of a tensor along a dimension with numerical stability.
    
    Args:
        tensor: Input tensor
        dim: Dimension to compute norm along
        eps: Small constant for numerical stability
        keepdim: Whether to keep the dimension
        
    Returns:
        L2 norm of tensor
    """
    return torch.sqrt(torch.sum(torch.square(tensor), dim=dim, keepdim=keepdim) + eps)

def get_local_rotation_frames(backbone_coords, sc1_coords=None, sc2_coords=None, fallback_vec=None):
    """
    Get local rotation frames either from backbone atoms or from sidechain atoms.
    For peptides, this function decides whether to use backbone or sidechain-defined frames.
    
    Args:
        backbone_coords: tensor of shape (B, L, 3, 3), ordered as [N, CA, C]
        sc1_coords: tensor of shape (B, L, 3), first sidechain atom (typically CB for amino acids)
        sc2_coords: tensor of shape (B, L, 3), optional second sidechain atom (typically CG)
        fallback_vec: optional fallback vector if sidechain atoms are missing or identical

    Returns:
        Rotation frame tensor of shape (B, L, 3, 3)
    """
    if sc1_coords is not None:
        # Use sidechain-defined frame (typically more specific to amino acid identity)
        return get_sidechain_rotation_frames(
            ca_coords=backbone_coords[:, :, 1],  # CA atom is at index 1
            sc1_coords=sc1_coords,
            sc2_coords=sc2_coords,
            fallback_vec=fallback_vec,
        )
    else:
        # Use backbone-defined frame (works for all amino acids including Glycine)
        return get_backbone_rotation_frames(backbone_coords)


def get_backbone_rotation_frames(coords):
    """
    Local frame using N, CA, C as origin and reference vectors.
    This is particularly useful for peptides as all amino acids have these backbone atoms.
    
    Args:
        coords: tensor of shape (B, L, 3, 3) with atom order [N, CA, C]

    Returns:
        Rotation frame tensor of shape (B, L, 3, 3) representing local coordinate system
    """
    v1 = coords[:, :, 2] - coords[:, :, 1]  # C - CA: First basis vector along CA-C bond
    v2 = coords[:, :, 0] - coords[:, :, 1]  # N - CA: Used to define plane for second basis vector
    
    # Gram-Schmidt process to create orthonormal basis
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * torch.sum(e1 * v2, dim=-1, keepdim=True)  # Orthogonalize v2 w.r.t e1
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)  # Third basis vector orthogonal to first two

    return torch.stack([e1, e2, e3], dim=-2)


def get_sidechain_rotation_frames(ca_coords, sc1_coords, sc2_coords=None, fallback_vec=None):
    """
    Local frame using CA as origin and sidechain atoms for directions.
    For peptides, this provides amino acid-specific frames since sidechain geometry varies.
    
    Args:
        ca_coords: (B, L, 3) - CA positions (alpha carbon)
        sc1_coords: (B, L, 3) - first sidechain atom (typically CB for amino acids except Gly)
        sc2_coords: (B, L, 3) - optional second sidechain atom (typically CG)
        fallback_vec: (3,) or (B, L, 3) - used if sc2_coords is None or for Glycine

    Returns:
        Rotation frame tensor of shape (B, L, 3, 3)
    """
    # First basis vector along CA-CB bond (or equivalent for non-standard amino acids)
    v1 = sc1_coords - ca_coords
    e1 = normalize(v1, dim=-1)

    if sc2_coords is not None:
        # Use second sidechain atom if available (e.g., CG)
        ref_vec = sc2_coords - ca_coords
    else:
        # Use fixed fallback vector if second atom not given (e.g., for Glycine)
        if fallback_vec is None:
            # Default direction along z-axis if not specified
            fallback_vec = torch.tensor([0.0, 0.0, 1.0], device=ca_coords.device, dtype=ca_coords.dtype)
        fallback_vec = fallback_vec.expand_as(v1)
        ref_vec = fallback_vec

    # Gram-Schmidt process to create orthonormal basis
    u2 = ref_vec - e1 * torch.sum(e1 * ref_vec, dim=-1, keepdim=True)  # Orthogonalize ref_vec w.r.t e1
    e2 = normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)  # Third basis vector orthogonal to first two

    return torch.stack([e1, e2, e3], dim=-2)


def rotate(v, R):
    """
    Rotates a vector v using a rotation matrix R.
    Used for transforming coordinates between global and local frames.
    
    Args:
        v: (..., 3) - vector to rotate
        R: (..., 3, 3) - rotation matrix
        
    Returns:
        Rotated vector (..., 3)
    """
    v = v.unsqueeze(-1)  # (..., 3, 1)
    return torch.sum(v * R, dim=-2)  # (..., 3)