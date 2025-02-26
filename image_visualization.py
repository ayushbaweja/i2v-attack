import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from diffusers import AutoencoderKLCogVideoX

def preprocess_image_for_vae(image_path, input_size=512, device="cpu"):

    image = Image.open(image_path).convert('RGB')
    # Scale pixel values to [-1, 1] range
    img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_array = img_array[:, :, :3]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
    ])
    
    image_tensor = transform(img_array).unsqueeze(0).to(device)
    return image_tensor

def encode_img(vae, image_tensor, dtype, device):
    vae.enable_slicing()
    vae.enable_tiling()
    
    transform = transforms.Compose([
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image_tensor).to(device).to(dtype)
    image_tensor = image_tensor.unsqueeze(2)  # Add time dimension
    
    with torch.no_grad():
        encoded_image = vae.encode(image_tensor).latent_dist.sample()
    
    return encoded_image

# RGB using PCA, Grayscale using mean of latents
def visualize_latent_space(images, vae, output_dir, device="cpu", dtype=torch.float32):
    print("Encoding images to latent space...")
    latents = {}
    
    for name, img_path in images.items():
        img_tensor = preprocess_image_for_vae(img_path, device=device)
        latent = encode_img(vae, img_tensor, dtype, device)
        
        latent_np = latent.squeeze(0).squeeze(1).cpu().numpy()
        latents[name] = latent_np
    
    # for grayscale
    mean_latents = {}
    for name, latent in latents.items():
        # mean across channels
        mean_latent = np.mean(latent, axis=0)
        
        # Normalizing
        mean_min = mean_latent.min()
        mean_max = mean_latent.max()
        mean_img = 255 * (mean_latent - mean_min) / (mean_max - mean_min)
        mean_img = mean_img.astype(np.uint8)
        
        mean_latents[name] = mean_img
    
    # Apply PCA to latent space (combine all latents first)
    print("Applying PCA to latent space...")
    
    # Flatten the channel dimension (first dimension) and spatial dimensions
    flattened_latents = {
        name: latent.reshape(latent.shape[0], -1).transpose()  # 4096x16 for 512x512
        for name, latent in latents.items()
    }
    
    # Stack all latents for fitting PCA
    all_latents = np.vstack([flattened_latents[name] for name in images.keys()])
    
    pca = PCA(n_components=3)
    pca.fit(all_latents)
    
    # Transform each latent separately
    pca_latents = {}
    for name, flat_latent in flattened_latents.items():
        pca_result = pca.transform(flat_latent)
        
        # Normalize to 0-255 range
        pca_min = pca_result.min(axis=0, keepdims=True)
        pca_max = pca_result.max(axis=0, keepdims=True)
        pca_result = 255 * (pca_result - pca_min) / (pca_max - pca_min)
        
        # Reshape to spatial dimensions (height, width, 3)
        h = w = int(np.sqrt(flat_latent.shape[0]))
        pca_image = pca_result.reshape(h, w, 3).astype(np.uint8)
        pca_latents[name] = pca_image
    
    # Visualize the latent space (both RGB PCA and grayscale mean)
    fig, axes = plt.subplots(2, len(images), figsize=(15, 10))
    
    # RGB PCA visualizations
    for i, (name, pca_img) in enumerate(pca_latents.items()):
        axes[0, i].imshow(pca_img)
        axes[0, i].set_title(f"{name.capitalize()} Latent PCA (RGB)")
        axes[0, i].axis('off')
    
    # Grayscale mean visualizations
    for i, (name, mean_img) in enumerate(mean_latents.items()):
        axes[1, i].imshow(mean_img, cmap='gray')
        axes[1, i].set_title(f"{name.capitalize()} Latent Mean (Gray)")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latent_space_comparison.png", dpi=300)
    plt.close()
    
    # Save individual images
    for name, pca_img in pca_latents.items():
        rgb_img = Image.fromarray(pca_img)
        rgb_img.save(f"{output_dir}/{name}_latent_pca_rgb.png")
        
    for name, mean_img in mean_latents.items():
        gray_img = Image.fromarray(mean_img)
        gray_img.save(f"{output_dir}/{name}_latent_mean_gray.png")
    
    print(f"Latent visualization complete! Results saved to {output_dir}/")
    return pca_latents, mean_latents

def main():
    parser = argparse.ArgumentParser(description="Visualize latent space representations using PCA")
    parser.add_argument("--clean_image", type=str, required=True,
                      help="Path to the clean input image")
    parser.add_argument("--perturbed_image", type=str, required=True,
                      help="Path to the perturbed image")
    parser.add_argument("--target_image", type=str, required=True,
                      help="Path to the target image")
    parser.add_argument("--input_size", type=int, default=512,
                      help="Input image size (default: 512)")
    parser.add_argument("--output_dir", type=str, default="./latent_visualization",
                      help="Directory for output files")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-2b",
                      help="Path to CogVideoX VAE model")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device and data type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load VAE model
    print("Loading VAE model...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_path, 
        subfolder="vae", 
        torch_dtype=dtype
    ).to(device)
    vae.eval()
    
    # Create a dictionary of image paths
    image_paths = {
        "clean": args.clean_image,
        "perturbed": args.perturbed_image,
        "target": args.target_image
    }
    
    # Visualize latent space
    visualize_latent_space(
        image_paths, vae, args.output_dir, device, dtype
    )

if __name__ == "__main__":
    main() 