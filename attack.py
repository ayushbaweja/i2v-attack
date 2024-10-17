from diffusers import AutoencoderKLCogVideoX
from PIL import Image
from torchvision import transforms
import torch
import argparse

def encode_img(vae, img_path, dtype, device):
    vae.enable_slicing()
    vae.enable_tiling()

    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device).to(dtype)

    with torch.no_grad():
        encoded_image = vae.encode(image_tensor).latent_dist.sample()
    
    return encoded_image

def main():
    parser = argparse.ArgumentParser(description="Encode an image using AutoencoderKLCogVideoX")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_path", type=str, default="encoded_image.pt", help="Path to save the encoded image tensor")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Using device: {device}")
    print(f"Loading VAE model...")
    vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=dtype).to(device)

    print(f"Encoding image: {args.image_path}")
    encoded_image = encode_img(vae, args.image_path, dtype, device)

    print(f"Saving encoded image to: {args.output_path}")
    torch.save(encoded_image, args.output_path)

    print("Encoding complete!")

if __name__ == "__main__":
    main()
