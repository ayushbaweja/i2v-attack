from diffusers import AutoencoderKLCogVideoX
from PIL import Image
from torchvision import transforms
import torch
import argparse

def preprocess_image(image_path, input_size=512):
    """
    Preprocess image following MIST approach.
    """
    image = Image.open(image_path).convert('RGB')
    # Scale pixel values to [-1, 1] range as done in MIST
    img_array = np.array(image).astype(np.float32) / 127.5 - 1.0
    img_array = img_array[:, :, :3]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((input_size, input_size)),
    ])
    
    image_tensor = transform(img_array).unsqueeze(0)
    return image_tensor

def encode_img(vae, img_path, dtype, device):
    vae.enable_slicing()
    vae.enable_tiling()
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device).to(dtype)
    image_tensor = image_tensor.unsqueeze(2)
    with torch.no_grad():
        encoded_image = vae.encode(image_tensor).latent_dist.sample()
    
    return encoded_image

class PGDAttack:
    def __init__(self, epsilon=0.062, alpha=0.001, num_steps=100):
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha     # Step size
        self.num_steps = num_steps
        self.loss_fn = nn.MSELoss()

    def attack(self, vae, clean_tensor, target_tensor, dtype, device):
        """
        Perform PGD attack to minimize distance between encoded representations.
        """
        # Initialize perturbed image as copy of clean image
        perturbed = clean_tensor.clone().detach().requires_grad_(True)
        
        for step in tqdm(range(self.num_steps), desc="Optimizing image"):
            if perturbed.grad is not None:
                perturbed.grad.zero_()
            
            # Forward pass
            perturbed_latent = encode_img(vae, perturbed, dtype, device)
            target_latent = encode_img(vae, target_tensor, dtype, device)
            
            # Compute loss
            loss = self.loss_fn(perturbed_latent, target_latent)
            
            # Backward pass
            loss.backward()
            
            # Update perturbed image
            with torch.no_grad():
                grad_sign = perturbed.grad.sign()
                perturbed.data = perturbed.data - self.alpha * grad_sign
                
                # Project back to epsilon ball and valid pixel range
                delta = torch.clamp(perturbed.data - clean_tensor, -self.epsilon, self.epsilon)
                perturbed.data = torch.clamp(clean_tensor + delta, -1, 1)
        
        return perturbed

def main():
    parser = argparse.ArgumentParser(description="i2v attack with VAE encodings")
    parser.add_argument("clean_image", type=str, help="Path to the clean input image")
    parser.add_argument("target_image", type=str, help="Path to the target image")
    parser.add_argument("--output_path", type=str, default="adversarial.png", help="Path to save the adversarial image")
    parser.add_argument("--epsilon", type=float, default=0.062, help="Maximum perturbation (default: 0.062)")
    parser.add_argument("--alpha", type=float, default=0.001, help="Step size for PGD (default: 0.001)")
    parser.add_argument("--steps", type=int, default=100, help="Number of PGD steps (default: 100)")
    parser.add_argument("--input_size", type=int, default=512, help="Input image size (default: 512)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device: {device}")

    # Load and preprocess images
    print("Preprocessing images...")
    clean_tensor = preprocess_image(args.clean_image, args.input_size)
    target_tensor = preprocess_image(args.target_image, args.input_size)

    # Load VAE model
    print("Loading VAE model...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-2b", 
        subfolder="vae", 
        torch_dtype=dtype
    ).to(device)
    vae.eval()

    # Perform attack
    print("Starting PGD attack...")
    attack = PGDAttack(
        epsilon=args.epsilon,
        alpha=args.alpha,
        num_steps=args.steps
    )
    adversarial_tensor = attack.attack(vae, clean_tensor, target_tensor, dtype, device)

    # Save result
    print(f"Saving adversarial image to: {args.output_path}")
    # Convert back to PIL image (reverse preprocessing)
    adversarial_np = (adversarial_tensor[0].cpu().detach().numpy() + 1) * 127.5
    adversarial_np = np.transpose(adversarial_np, (1, 2, 0))
    adversarial_image = Image.fromarray(adversarial_np.astype(np.uint8))
    adversarial_image.save(args.output_path)

    print("Attack complete!")

if __name__ == "__main__":
    main()
