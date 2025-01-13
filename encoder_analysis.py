import argparse
import torch 
import numpy as np
from diffusers import AutoencoderKLCogVideoX
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def preprocess_image(image_path, input_size=512, device="cpu"):
    image = Image.open(image_path).convert('RGB')
    # Scale pixel values to [-1, 1] range as done in MIST
    img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_array = img_array[:, :, :3]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
    ])

    # batch, channel, time, height, width
    # Create tensor [1, 3, 1, H, W] - adding batch and time dimensions
    image_tensor = transform(img_array).unsqueeze(0).unsqueeze(2).to(device)
    return image_tensor

def encode_image(model, image_tensor, dtype):
    image_tensor = image_tensor.to(dtype)
    with torch.no_grad():
        encoded = model.encode(image_tensor)[0].sample() # tuple is (latent dist, intermediate state). we sample gaussian to get latent values
    return encoded

def analyze_latents(clean_latent, perturbed_latent, target_latent, output_dir="./"):
    print("\n Clean latent shape: ", clean_latent.shape)

    flat_latents = [] #1d
    for latent in [clean_latent, perturbed_latent, target_latent]:
        # removing batch and time
        flat = latent.squeeze(0).squeeze(1).reshape(-1).to(torch.float32).cpu().numpy()
        flat_latents.append(flat)
    
    # stack for pca
    # 16x64x64, 16 is latent and spatial dim reduced by factor 8 so 64x64
    all_latents = np.vstack(flat_latents)

    pca = PCA(n_components=3)
    latents_3d = pca.fit_transform(all_latents)

    # 3d viz
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    labels = ['Clean', 'Perturbed', 'Target']
    colors = ['blue', 'green', 'red']
    markers = ['o', '^', 's']

    for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        ax.scatter(latents_3d[i, 0], 
                  latents_3d[i, 1], 
                  latents_3d[i, 2],
                  c=color, marker=marker, s=100, label=label)
    
    ax.set_title('VAE Latent Space Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/latent_space_3d.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze VAE latent representations")
    parser.add_argument("--clean_image", type=str, required=True,
                      help="Path to the clean input image")
    parser.add_argument("--perturbed_image", type=str, required=True,
                      help="Path to the perturbed image")
    parser.add_argument("--target_image", type=str, required=True,
                      help="Path to the target image")
    parser.add_argument("--model_path", type=str, required=True,
                      help="Path to CogVideoX VAE model")
    parser.add_argument("--input_size", type=int, default=512,
                      help="Input image size (default: 512)")
    parser.add_argument("--output_dir", type=str, default="./encoder_analysis",
                      help="Directory for output files")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                      help="Data type for computation (float16 or bfloat16)")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load VAE model
    print("Loading VAE model...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=dtype
    ).to(device)
    
    # Enable optimizations
    vae.enable_slicing()
    vae.enable_tiling()
    vae.eval()

    # Process images
    print("Processing images...")
    images = {
        "clean": preprocess_image(args.clean_image, args.input_size, device),
        "perturbed": preprocess_image(args.perturbed_image, args.input_size, device),
        "target": preprocess_image(args.target_image,args.input_size, device)
    }
    
    # Get latents
    print("Getting encoder latents...")
    latents = {name: encode_image(vae, img, dtype) 
              for name, img in images.items()}
    
    # Analyze latents
    print("Analyzing latent space...")
    analyze_latents(
        latents["clean"],
        latents["perturbed"],
        latents["target"],
        args.output_dir
    )

if __name__ == "__main__":
    main()

