import torch
import clip
from PIL import Image
from torchvision import transforms
import os
from glob import glob
import argparse
import numpy as np

# CLIP 모델과 변환기 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_score(image, text, model, preprocess, device):
    """
    Compute the CLIP score between an image and a text prompt.
    
    Args:
        image (PIL.Image or torch.Tensor): The input image.
        text (str): The input text prompt.
        model (clip.Model): The CLIP model.
        preprocess (torchvision.transforms): Preprocessing function for images.
        device (str): Device to run the model on.
    
    Returns:
        float: The CLIP score.
    """
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    
    # Preprocess the image and text
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)
    
    # Compute the features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
    
    # Normalize the features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # Compute the cosine similarity
    similarity = (image_features @ text_features.T).item()
    return similarity

def load_images_from_folder(folder_path):
    """
    Load images from a specified folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
    
    Returns:
        List of PIL.Images.
    """
    image_files = sorted(glob(os.path.join(folder_path, '*.png')))
    images = [Image.open(f).convert("RGB") for f in image_files]
    return images

def main(folder_path, text_prompt):
    images = load_images_from_folder(folder_path)
    if not images:
        print("No images found in the folder.")
        return
    
    clip_scores = []
    
    for image in images:
        clip_score = compute_clip_score(image, text_prompt, model, preprocess, device)
        clip_scores.append(clip_score)
    
    avg_clip_score = np.mean(clip_scores)
    print("Average CLIP Score:", avg_clip_score)

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
main(args.folder_path, args.text_prompt)