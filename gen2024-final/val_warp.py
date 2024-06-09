import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from glob import glob
import argparse
import warnings

# 모든 경고 무시
warnings.filterwarnings("ignore")

def compute_warp_error(flow, frame1, frame2):
    """
    Compute the warp error between two frames given the optical flow.
    
    Args:
        flow (torch.Tensor): Optical flow from frame1 to frame2. Shape: (N, 2, H, W)
        frame1 (torch.Tensor): First frame. Shape: (N, C, H, W)
        frame2 (torch.Tensor): Second frame. Shape: (N, C, H, W)
    
    Returns:
        torch.Tensor: Warp error. Shape: (N,)
    """
    N, C, H, W = frame1.shape
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
    grid_x = grid_x.to(flow.device).float()
    grid_y = grid_y.to(flow.device).float()
    
    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]
    
    warped_grid_x = grid_x.unsqueeze(0) + u
    warped_grid_y = grid_y.unsqueeze(0) + v
    
    warped_grid_x = 2.0 * warped_grid_x / (W - 1) - 1.0
    warped_grid_y = 2.0 * warped_grid_y / (H - 1) - 1.0
    
    grid = torch.stack((warped_grid_x, warped_grid_y), dim=-1)
    
    warped_frame1 = F.grid_sample(frame1, grid, mode='bilinear', padding_mode='border')
    
    warp_error = torch.mean((warped_frame1 - frame2) ** 2, dim=[1, 2, 3])
    return warp_error

def load_frames_from_folder(folder_path):
    """
    Load frames from a specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
    
    Returns:
        List of frames.
    """
    frame_files = sorted(glob(os.path.join(folder_path, '*.png')))
    frames = [cv2.imread(f) for f in frame_files]
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    frames = [torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0 for frame in frames]
    return frames

def compute_optical_flow(frame1, frame2):
    """
    Compute the optical flow between two frames using OpenCV's Farneback method.
    
    Args:
        frame1 (np.ndarray): First frame.
        frame2 (np.ndarray): Second frame.
    
    Returns:
        np.ndarray: Optical flow.
    """

    gray1 = cv2.cvtColor(frame1.transpose(1,2,0), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2.transpose(1,2,0), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
    return flow

def main(folder_path):
    frames = load_frames_from_folder(folder_path)
    num_frames = len(frames)
    
    if num_frames < 2:
        print("Not enough frames to compute warp error.")
        return
    
    warp_errors = []
    
    for i in range(num_frames - 1):
        frame1 = frames[i].unsqueeze(0)
        frame2 = frames[i + 1].unsqueeze(0)
        
        flow = compute_optical_flow(frame1.squeeze().numpy(), frame2.squeeze().numpy())

        warp_error = compute_warp_error(flow, frame1, frame2)
        warp_errors.append(warp_error.item())
    
    avg_warp_error = np.mean(warp_errors)
    print("Average Warp Error:", avg_warp_error)

def parse_arg():
    parser = argparse.ArgumentParser(description='Compute CLIP score for images in a folder.')
    parser.add_argument('--folder_path', type=str, 
        default='tokenflow-results_pnp_SD_2.1/wolf/a shiny silver robotic wolf', 
        help='Path to the folder containing images.')
    return parser.parse_args()

# Example usage
args = parse_arg()
if 'pnp' in args.folder_path:
    args.folder_path += '/attn_0.5_f_0.8/batch_size_8/50/img_ode'
else:
    args.folder_path += '/batch_size_8/50start_0.9/img_ode'
args.text_prompt = args.folder_path.split('/')[-1]
# text_prompt = "A description of the images"  # Replace with your text prompt
main(args.folder_path)