from diffusers import AutoencoderKLCogVideoX
from PIL import Image
from torchvision import transforms
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm

def preprocess_image(image_path, input_size=512, device="cpu"):
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
    
    image_tensor = transform(img_array).unsqueeze(0).to(device)
    return image_tensor

def encode_img(vae, image_tensor, dtype, device):
    vae.enable_slicing()
    vae.enable_tiling()
    
    transform = transforms.Compose([
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image_tensor).to(device).to(dtype)
    image_tensor = image_tensor.unsqueeze(2)
    
    with torch.enable_grad():
        encoded_image = vae.encode(image_tensor).latent_dist.sample()
    
    return encoded_image

class EncoderNoiseAnalysis:
    def __init__(self, vae, noise_scales=[0.1, 0.2, 0.5, 1.0]):
        self.vae = vae
        self.noise_scales = noise_scales
        self.loss_fn = nn.MSELoss()
    
    def analyze_sensitivity(self, clean_latent, device, dtype):
        sensitivities = {}
        latent_shape = clean_latent.shape
        for scale in self.noise_scales:
            noise = torch.randn_like(clean_latent) * scale
            noisy_latent = clean_latent + noise

            with torch.no_grad():
                clean_recon = self.vae.decode(clean_latent).sample
                noisy_recon = self.vae.decode(noisy_latent).sample
            
            recon_loss = self.loss_fn(clean_recon, noisy_recon)
            sensitivities[scale] = recon_loss.item()
        return sensitivities

class PGDAttack:
    def __init__(self, epsilon=0.062, alpha=0.001, num_steps=100, analyze_noise=False):
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha     # Step size
        self.num_steps = num_steps
        self.loss_fn = nn.MSELoss()
        self.analyze_noise = analyze_noise

    def attack(self, vae, clean_tensor, target_tensor, dtype, device):
        """
        Perform PGD attack to minimize distance between encoded representations.
        """
        # Initialize perturbed image as copy of clean image
        perturbed = clean_tensor.clone().detach().requires_grad_(True)

        if self.analyze_noise:
            noise_analyzer = EncoderNoiseAnalysis(vae)
        
        loss_history = []
        sensitivity_history = []
        
        for step in tqdm(range(self.num_steps), desc="Optimizing image"):
            if perturbed.grad is not None:
                perturbed.grad.zero_()
            
            # Forward pass
            perturbed_latent = encode_img(vae, perturbed, dtype, device)
            target_latent = encode_img(vae, target_tensor, dtype, device)
            
            # Compute loss
            loss = self.loss_fn(perturbed_latent, target_latent)
            loss_history.append(loss.item())

            if self.analyze_noise and step % 10 == 0:
                sensitivities = noise_analyzer.analyze_sensitivity(perturbed_latent, device, dtype)
                sensitivity_history.append(sensitivities)
            
            # Backward pass
            loss.backward()
            
            # Update perturbed image
            with torch.enable_grad():
                grad_sign = perturbed.grad.sign()
                perturbed.data = perturbed.data - self.alpha * grad_sign
                
                # Project back to epsilon ball and valid pixel range
                delta = torch.clamp(perturbed.data - clean_tensor, -self.epsilon, self.epsilon)
                perturbed.data = torch.clamp(clean_tensor + delta, -1, 1)
        
        results = {
            'perturbed': perturbed,
            'loss_history': loss_history
        }
        
        if self.analyze_noise:
            results['sensitivity_history'] = sensitivity_history
            
        return results

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
    #print("CUDA Available:", torch.cuda.is_available())
    #print("CUDA Device Count:", torch.cuda.device_count())
    #print("Current Device:", device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Using device: {device}")

    # Load and preprocess images
    print("Preprocessing images...")
    clean_tensor = preprocess_image(args.clean_image, args.input_size, device)
    target_tensor = preprocess_image(args.target_image, args.input_size, device)

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
        num_steps=args.steps,
        analyze_noise=True
    )
    results = attack.attack(vae, clean_tensor, target_tensor, dtype, device)

    if results['sensitivity_history']:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        for scale in results['sensitivity_history'][0].keys():
            sensitivities = [h[scale] for h in results['sensitivity_history']]
            plt.plot(sensitivities, label=f'Noise scale {scale}')
        
        plt.xlabel('Attack step (x10)')
        plt.ylabel('Reconstruction Loss')
        plt.title('Encoder Noise Sensitivity During Attack')
        plt.legend()
        plt.savefig('noise_sensitivity.png')

    # Save result
    print(f"Saving adversarial image to: {args.output_path}")
    # Convert back to PIL image (reverse preprocessing)
    adversarial_np = (results['perturbed'][0].cpu().detach().numpy() + 1) * 127.5
    adversarial_np = np.transpose(adversarial_np, (1, 2, 0))
    adversarial_image = Image.fromarray(adversarial_np.astype(np.uint8))
    adversarial_image.save(args.output_path)

    print("Attack complete!")

if __name__ == "__main__":
    main()
