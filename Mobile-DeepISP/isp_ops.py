import torch
import torch.nn.functional as F

def apply_white_balance(raw_tensor, wb_gains):
    B, C, H, W = raw_tensor.shape
    assert C == 1, "Expected single-channel RAW input"
    
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij', device=raw_tensor.device)
    r_mask = ((ys % 2 == 0) & (xs % 2 == 0)).unsqueeze(0).unsqueeze(0)
    g_mask = (((ys % 2 == 0) & (xs % 2 == 1)) | ((ys % 2 == 1) & (xs % 2 == 0))).unsqueeze(0).unsqueeze(0)
    b_mask = ((ys % 2 == 1) & (xs % 2 == 1)).unsqueeze(0).unsqueeze(0)

    R_gain, G_gain, B_gain = wb_gains
    balanced = raw_tensor * (r_mask * R_gain + g_mask * G_gain + b_mask * B_gain)
    return balanced

def demosaic(raw_tensor):
    B, C, H, W = raw_tensor.shape
    assert C == 1, "Expected single-channel RAW for demosaicing."
    device = raw_tensor.device
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij', device=device)
    r_mask = ((ys % 2 == 0) & (xs % 2 == 0))
    g_mask = (((ys % 2 == 0) & (xs % 2 == 1)) | ((ys % 2 == 1) & (xs % 2 == 0)))
    b_mask = ((ys % 2 == 1) & (xs % 2 == 1))

    R = torch.where(r_mask, raw_tensor.squeeze(1), torch.zeros_like(raw_tensor.squeeze(1)))
    G = torch.where(g_mask, raw_tensor.squeeze(1), torch.zeros_like(raw_tensor.squeeze(1)))
    B = torch.where(b_mask, raw_tensor.squeeze(1), torch.zeros_like(raw_tensor.squeeze(1)))

    kernel = torch.ones((1,1,3,3), device=device) / 4.0
    R_filled = F.conv2d(R.unsqueeze(1), kernel, padding=1)
    R = torch.where(r_mask.unsqueeze(0).unsqueeze(0), R.unsqueeze(1), R_filled)

    G_filled = F.conv2d(G.unsqueeze(1), kernel, padding=1)
    G = torch.where(g_mask.unsqueeze(0).unsqueeze(0), G.unsqueeze(1), G_filled)

    B_filled = F.conv2d(B.unsqueeze(1), kernel, padding=1)
    B = torch.where(b_mask.unsqueeze(0).unsqueeze(0), B.unsqueeze(1), B_filled)

    rgb = torch.cat([R, G, B], dim=1)
    return rgb.clamp(0.0, 1.0)

def color_correction(rgb, cc_matrix):
    B, C, H, W = rgb.shape
    flat = rgb.view(B, 3, -1)
    corrected = torch.matmul(cc_matrix.to(rgb.device), flat)
    corrected = corrected.view(B,3,H,W)
    return corrected.clamp(0.0,1.0)

def tone_mapping(rgb):
    return 3*(rgb**2) - 2*(rgb**3)

def apply_gamma(rgb):
    low_mask = rgb <= 0.0031308
    rgb[low_mask] = 12.92 * rgb[low_mask]
    rgb[~low_mask] = 1.055 * (rgb[~low_mask]**(1.0/2.4)) - 0.055
    return rgb.clamp(0.0, 1.0)
