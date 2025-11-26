import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile # Using scipy for audio saving
import torch.optim as optim # Import optimizer

# Placeholder classes (as defined previously)
class RiemannianVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class GeodesicAttention(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

class MoEFeedForward(nn.Module):
    def __init__(self, latent_dim, num_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        ) for _ in range(num_experts)])
        self.gate = nn.Linear(latent_dim, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        expert_weights = F.softmax(gate_logits, dim=-1)
        
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=-1)
        
        output = (outputs * expert_weights.unsqueeze(-2)).sum(dim=-1)
        return output

# ==========================================
# 1. NEW COMPONENT: Neural Kuramoto ODE Layer
# ==========================================
class KuramotoInteraction(nn.Module):
    """
    Vectorized Kuramoto Oscillator coupling.
    Stores 'last_attn_weights' for visualization.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.omega = nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.to_coupling = nn.Linear(dim, dim * 2, bias=False)
        self.output = nn.Linear(dim, dim)
        
        # Buffer to store weights for visualization
        self.last_attn_weights = None

    def forward(self, t, z):
        B, T, D = z.shape
        intrinsic_drift = self.omega 
        
        q, k = self.to_coupling(z).chunk(2, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        q = q.permute(0, 2, 1, 3) # [B, H, T, D_h]
        k = k.permute(0, 2, 1, 3)
        
        coupling = torch.matmul(q, k.transpose(-2, -1)) # [B, H, T, T]
        
        self.last_attn_weights = coupling.detach().cpu()
        
        interaction = torch.sin(coupling) 
        force = torch.matmul(interaction, k)
        force = force.permute(0, 2, 1, 3).reshape(B, T, D)
        
        return intrinsic_drift + self.output(force)

# ==========================================
# 2. UPDATED COMPONENT: Geometric ODE Solver
# ==========================================
class GeometricDynamicsPredictor(nn.Module):
    """ 
    Replaces standard Neural ODE. 
    [cite_start]Uses Kuramoto logic to model 'smooth neural state evolution'[cite: 22].
    """
    def __init__(self, dim):
        super().__init__()
        self.func = KuramotoInteraction(dim)
        
    def forward(self, z):
        dt = 0.25
        t = 0.0
        x = z
        
        for _ in range(4):
            x = F.normalize(x, p=2, dim=-1) 
            
            k1 = self.func(t, x)
            k2 = self.func(t + dt/2, x + dt/2 * k1)
            k3 = self.func(t + dt/2, x + dt/2 * k2)
            k4 = self.func(t + dt, x + dt * k3)
            
            x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            
        return F.normalize(x, p=2, dim=-1)

# ==========================================
# 3. UPDATED MODEL: ManifoldFormer
# ==========================================
class ManifoldFormer(nn.Module):
    def __init__(self, input_dim=4800, latent_dim=128):
        super().__init__()
        self.vae = RiemannianVAE(input_dim, latent_dim)
        
        self.ln1 = nn.LayerNorm(latent_dim)
        self.attn = GeodesicAttention(latent_dim)
        self.ln2 = nn.LayerNorm(latent_dim)
        self.moe = MoEFeedForward(latent_dim)
        
        self.ode = GeometricDynamicsPredictor(latent_dim)
        
        self.head = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        recon_vae, z = self.vae(x)
        
        z_trans = z + self.attn(self.ln1(z))
        z_trans = z_trans + self.moe(self.ln2(z_trans))
        
        z_dyn = self.ode(z_trans)
        
        out = self.head(z_dyn)
        
        return out, recon_vae, z, z_trans, z_dyn

# ==========================================
# 4. UPDATED LOSS: Temporal Geometric Loss
# ==========================================
def temporal_geometric_loss(pred, target, z_latent, z_dyn, alpha=1.0, beta=0.1, gamma=0.01):
    """
    Combines Reconstruction, Manifold Alignment, and Temporal Stability.
    """
    l_recon = F.mse_loss(pred, target)
    
    idx = torch.randperm(target.size(0))
    x_flat = target.mean(dim=1)
    z_flat = z_latent.mean(dim=1)
    d_input = torch.norm(x_flat - x_flat[idx], p=2, dim=-1)
    dot = (z_flat * z_flat[idx]).sum(dim=-1).clamp(-0.99, 0.99)
    d_geo = torch.acos(dot)
    l_geo = F.mse_loss(d_geo, d_input)
    
    norm_violation = (torch.norm(z_dyn, p=2, dim=-1) - 1.0).pow(2).mean()
    
    l_smooth = F.mse_loss(z_dyn[:, 1:, :], z_dyn[:, :-1, :])
    
    return l_recon + alpha*l_geo + beta*norm_violation + gamma*l_smooth

def visualize_neural_synchronization(model, input_batch):
    """
    Runs a forward pass and visualizes the Kuramoto dynamics, saving plots to files.
    """
    model.eval()
    with torch.no_grad():
        _, _, _, _, z_dyn = model(input_batch)
        
        attn_weights = model.ode.func.last_attn_weights
        
        latent_phases = z_dyn.cpu()

    plt.figure(figsize=(6, 5))
    sync_map = attn_weights[0, 0, :, :].numpy() 
    
    plt.imshow(sync_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label="Coupling Strength (Phase Sync)")
    plt.title("Neural Phase Synchronization\n(Time x Time Coupling)")
    plt.xlabel("Target Time Step")
    plt.ylabel("Source Time Step")
    plt.tight_layout()
    plt.savefig("sync_map.png")
    plt.close()
    
    print("Saved 'sync_map.png'")

    plt.figure(figsize=(6, 5))
    
    c = latent_phases[0][:, 0]
    s = latent_phases[0][:, 1]
    angles = np.arctan2(s, c)
    radii = np.ones_like(angles)
    
    colors = np.linspace(0, 1, len(angles)) # Define colors here
    plt.subplot(projection='polar')
    plt.scatter(angles, radii, c=colors, cmap='plasma', alpha=0.75, s=50)
    plt.title("Latent Oscillator Phases\n(Color = Time Evolution)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("phase_portrait.png")
    plt.close()

    print("Saved 'phase_portrait.png'")

class SyntheticChordsDataset(Dataset):
    """
    Generates synthetic 'music' consisting of moving chords.
    Perfect for testing if Kuramoto dynamics can lock onto frequencies.
    """
    def __init__(self, num_samples=1000, seq_len=128, channels=64):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        t = np.linspace(0, 4*np.pi, self.seq_len)
        
        freqs = np.random.choice([1, 2, 3, 5, 8], size=3, replace=False)
        
        signal = np.zeros((self.seq_len, self.channels))
        
        for ch in range(self.channels):
            phase_shift = (ch / self.channels) * 2 * np.pi
            wave = (np.sin(freqs[0]*t + phase_shift) + 
                    0.5 * np.sin(freqs[1]*t + phase_shift) + 
                    0.25 * np.sin(freqs[2]*t + phase_shift))
            
            signal[:, ch] = wave + np.random.normal(0, 0.1, size=len(t))
            
        return torch.FloatTensor(signal), torch.FloatTensor(signal) # Input, Target

# Removed GTZANMelDataset (as it's commented out and torchaudio is no longer used)
# Removed torchaudio and os imports

# Helper function to save audio
def save_audio(tensor, filename, sample_rate=22050):
    """
    Saves a given tensor as an audio file using scipy.io.wavfile.
    Assumes tensor is [Time, Channels] or [Time].
    If multichannel, saves first channel for simplicity.
    Normalizes to -1 to 1 and converts to 16-bit PCM.
    """
    if tensor.ndim == 2: # [Time, Channels]
        audio_data = tensor[:, 0] # Take first channel
    else: # [Time]
        audio_data = tensor
    
    # Move to CPU and convert to numpy
    audio_data_np = audio_data.cpu().numpy()

    # Normalize to -1 to 1 range (assuming float input)
    audio_data_np = audio_data_np / np.max(np.abs(audio_data_np))
    
    # Convert to 16-bit PCM for WAV file
    audio_data_np = (audio_data_np * 32767).astype(np.int16)

    wavfile.write(filename, sample_rate, audio_data_np)
    print(f"Saved audio to {filename}")


def main():
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 2. Synthetic Dataset and DataLoader
    seq_len = 128
    channels = 64
    synth_ds = SyntheticChordsDataset(num_samples=100, seq_len=seq_len, channels=channels)
    synth_loader = DataLoader(synth_ds, batch_size=4, shuffle=True)

    # 3. Model, Optimizer, Loss Function
    model = ManifoldFormer(input_dim=channels, latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    print(f"\n--- Starting Training for {num_epochs} Epochs ---")

    # For saving a sample after training
    sample_input, sample_target = next(iter(synth_loader))
    sample_input = sample_input[0:1].to(device) # Take one sample from the batch
    sample_target = sample_target[0:1] # Keep on CPU for saving
    
    # Save original sample audio
    save_audio(sample_target[0].cpu(), "original_audio_sample.wav", sample_rate=22050)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (input_batch, target_batch) in enumerate(synth_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            out, recon_vae, z_latent, z_trans, z_dyn = model(input_batch)
            
            loss = temporal_geometric_loss(out, target_batch, z_latent, z_dyn)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(synth_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    print("\n--- Training Complete ---")

    # 4. Generate and save audio sample after training
    model.eval()
    with torch.no_grad():
        generated_output, _, _, _, _ = model(sample_input)
        # Assuming output is [1, Time, Channels]
        generated_audio = generated_output[0].cpu() # Get the first sample, move to CPU
        save_audio(generated_audio, "generated_audio_epoch_10.wav", sample_rate=22050)
    
    # 5. Run visualization for a trained model output
    print("\n--- Displaying visualization for a trained model output ---")
    visualize_neural_synchronization(model, sample_input)
    

if __name__ == '__main__':
    main()