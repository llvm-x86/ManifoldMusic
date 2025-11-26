import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training import train_state
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import optax
import sys

# --- Flax Model Components ---

class VAE(nn.Module):
    input_dim: int
    latent_dim: int

    def setup(self):
        # Encoder: Causal Convolution to capture temporal context (phase/velocity)
        self.encoder_conv1 = nn.Conv(features=64, kernel_size=(5,), padding='CAUSAL')
        self.encoder_conv2 = nn.Conv(features=64, kernel_size=(5,), padding='CAUSAL')
        self.encoder_dense = nn.Dense(self.latent_dim)
        
        # Decoder: MLP to map from manifold back to signal space
        self.decoder_hidden = nn.Dense(128)
        self.decoder_out = nn.Dense(self.input_dim)

    def __call__(self, x):
        # Encoder
        h = nn.relu(self.encoder_conv1(x))
        h = nn.relu(self.encoder_conv2(h))
        z_raw = self.encoder_dense(h)
        
        norm = jnp.linalg.norm(z_raw, axis=-1, keepdims=True)
        z = z_raw / jnp.maximum(norm, 1e-8)
        
        # Decoder (Pointwise)
        recon = self.decode(z)
        
        return recon, z

    def decode(self, z):
        h_dec = nn.relu(self.decoder_hidden(z))
        return self.decoder_out(h_dec)

class Dynamics(nn.Module):
    latent_dim: int
    memory_gamma: float = 0.5  # Power-law decay parameter
    memory_kernel_size: int = 256

    @nn.compact
    def __call__(self, z):
        # z has shape (batch, seq_len, latent_dim)
        B, T, D = z.shape

        # 1. Power-Law Memory Integral (Causal Convolution)
        # Pre-calculate the memory force field.
        kernel = self.param(
            'memory_kernel',
            lambda key, shape, dtype: get_fractional_kernel(self.memory_gamma, self.memory_kernel_size),
            (self.memory_kernel_size, 1, 1), jnp.float32
        )
        memory_force = nn.Conv(
            features=self.latent_dim,
            kernel_size=(self.memory_kernel_size,),
            feature_group_count=self.latent_dim,
            padding='CAUSAL'
        )(z)

        # 2. Prepare Kuramoto Components (Keys/Values)
        # We treat the input sequence 'z' as the background field for the attention.
        # Ideally, for RK4, the field should be dynamic, but for efficiency in parallel training,
        # we assume the 'Keys' and 'Values' are fixed by the input trajectory,
        # and only the 'Query' (the particle being evolved) changes during the RK4 substeps.
        
        k_proj_layer = nn.Dense(self.latent_dim, name="K_proj")
        q_proj_layer = nn.Dense(self.latent_dim, name="Q_proj")
        out_proj_layer = nn.Dense(self.latent_dim, name="Out_proj")

        k = k_proj_layer(z)
        k_hat = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-8)
        
        # Causal Mask for Attention
        # mask[i, j] = 1 if i >= j else 0
        mask = jnp.tril(jnp.ones((T, T)))
        mask = mask[None, :, :] # (1, T, T)

        # Define the Vector Field Function for RK4
        # z_curr: (B, T, D) - The current estimate of the state at each time step
        def vector_field(z_curr):
            # Project z_curr to Query
            q = q_proj_layer(z_curr)
            q_hat = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
            
            # Causal Attention: Query (Current) vs Keys (History/Context)
            dots = jnp.einsum('btd,bTd->btT', q_hat, k_hat)
            
            # Apply Mask: Set future positions to -inf before softmax? 
            # But this is Kuramoto (sin), not Softmax. 
            # We just zero out the contributions from the future.
            attn_weights = jnp.sin(dots) * mask
            
            # Compute Force
            kuramoto_force_raw = jnp.einsum('btT,bTd->btd', attn_weights, k_hat)
            kuramoto_force = out_proj_layer(kuramoto_force_raw)
            
            total_force = memory_force + kuramoto_force
            
            # Project to Tangent Space of z_curr
            dot_prod = jnp.sum(total_force * z_curr, axis=-1, keepdims=True)
            tangent_vel = total_force - dot_prod * z_curr
            return tangent_vel

        # 3. Projected RK4 Integration
        h = 1.0 # Step size (1 time step)
        
        # Retraction Operator
        def retract(z_base, v):
            z_new = z_base + v
            return z_new / jnp.maximum(jnp.linalg.norm(z_new, axis=-1, keepdims=True), 1e-8)

        # RK4 Steps
        # k1 = f(z)
        k1 = vector_field(z)
        
        # k2 = f(Retract(z, h/2 * k1))
        z_k2 = retract(z, 0.5 * h * k1)
        k2 = vector_field(z_k2)
        
        # k3 = f(Retract(z, h/2 * k2))
        z_k3 = retract(z, 0.5 * h * k2)
        k3 = vector_field(z_k3)
        
        # k4 = f(Retract(z, h * k3))
        z_k4 = retract(z, h * k3)
        k4 = vector_field(z_k4)
        
        # Combine
        v_final = (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Final Retraction
        z_next = retract(z, v_final)
        
        return z_next

class ManifoldFormer(nn.Module):
    input_dim: int
    latent_dim: int

    def setup(self):
        self.vae = VAE(self.input_dim, self.latent_dim)
        self.dynamics = Dynamics(self.latent_dim)
        self.iso_scale = self.param('iso_scale', nn.initializers.ones, (1,))

    def __call__(self, x):
        recon_vae, z = self.vae(x)
        z_dyn = self.dynamics(z)
        
        # Decode the dynamic state
        out = self.vae.decode(z_dyn)
        
        return out, recon_vae, z, z_dyn, self.iso_scale

# --- Loss Function ---

def loss_fn(params, state, x, y):
    pred, recon_vae, z, z_dyn, iso_scale = state.apply_fn({'params': params}, x)
    
    l_recon = jnp.mean((pred - y) ** 2)
    l_recon_vae = jnp.mean((recon_vae - x) ** 2)
    
    # Geodesic smoothness on the sphere
    dot_product = jnp.sum(z_dyn * z, axis=-1).clip(-1.0 + 1e-6, 1.0 - 1e-6)
    geodesic_dist = jnp.arccos(dot_product)
    l_smooth = jnp.mean(geodesic_dist ** 2)
    
    # Scaled Isometry Loss
    # Input distances (Euclidean)
    dist_x = jnp.sqrt(jnp.sum((x[:, :, None, :] - x[:, None, :, :]) ** 2, axis=-1) + 1e-8)
    
    # Latent distances (Geodesic)
    dot_z = jnp.einsum('btd,bTd->btT', z, z).clip(-1.0 + 1e-6, 1.0 - 1e-6)
    dist_z = jnp.arccos(dot_z)
    
    l_iso = jnp.mean((iso_scale * dist_z - dist_x) ** 2)
    
    return l_recon + l_recon_vae + 0.1 * l_smooth + 0.01 * l_iso

# --- Data Generation ---

class SyntheticChordsDataset:
    def __init__(self, num_samples=1000, seq_len=128, channels=64, batch_size=16):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.batch_size = batch_size
        
    def __len__(self):
        return self.num_samples // self.batch_size
    
    def __getitem__(self, idx):
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        
        signals = []
        for _ in range(self.batch_size):
            freqs = np.random.choice([1, 2, 3, 5, 8], size=3, replace=False)
            signal = np.zeros((self.seq_len, self.channels))
            
            for ch in range(self.channels):
                phase_shift = (ch / self.channels) * 2 * np.pi
                wave = (np.sin(freqs[0] * t + phase_shift) + 
                        0.5 * np.sin(freqs[1] * t + phase_shift) + 
                        0.25 * np.sin(freqs[2] * t + phase_shift))
                signal[:, ch] = wave + np.random.normal(0, 0.1, size=len(t))
            signals.append(signal)
            
        signals = np.array(signals)
        return signals[:, :-1, :], signals[:, 1:, :]

# --- Audio Saving ---

def get_fractional_kernel(gamma, kernel_size, device=None):
    """
    Generates the discrete approximation of the fractional integral kernel
    (t-tau)^(-gamma) / Gamma(1-gamma) for a causal convolution.
    """
    t = jnp.arange(1, kernel_size + 1, dtype=jnp.float32)
    
    # Using JAX's lgamma for the log of the Gamma function
    gamma_val = jnp.exp(jax.lax.lgamma(1 - jnp.array(gamma)))
    weights = (1.0 / gamma_val) * (t ** (-gamma))
    
    # Reshape for conv1d: (Out_channels, In_channels, Time) -> (Length, In, Out) for Flax
    return weights[::-1].reshape(kernel_size, 1, 1)


def save_audio(tensor, filename, sample_rate=22050):
    audio_data = tensor[:, 0] if tensor.ndim == 2 else tensor
    audio_data_np = np.array(audio_data)
    audio_data_np /= np.max(np.abs(audio_data_np))
    audio_data_np = (audio_data_np * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, audio_data_np)
    print(f"Saved audio to {filename}")

# --- Imputation ---

def geodesic_imputation(model, params, sequence_with_gap, gap_start, gap_end, key, epsilon=1e-7):
    """
    Imputes a gap using the learned Manifold ODE dynamics (Autoregressive Rollout).
    This follows the 'geodesic' flow of the dynamical system, preserving phase and frequency.
    """
    original_seq_len = sequence_with_gap.shape[0]

    # 1. Encode the full sequence to get the context before the gap
    _, z_full_latent = model.apply({'params': params}, sequence_with_gap, method=lambda module, x: module.vae(x))
    
    # We start with the valid history up to the gap
    # Shape: (1, T, D) - adding batch dim for the model
    current_z_seq = z_full_latent[:gap_start][None, ...] 
    
    # We need to generate up to the end of the shadow region (to fix the encoder corruption)
    # The encoder receptive field is ~16.
    receptive_field = 16
    fill_end = min(gap_end + receptive_field, original_seq_len)
    steps_to_generate = fill_end - gap_start
    
    print(f"Autoregressively generating {steps_to_generate} steps using learned dynamics...")
    
    # JIT compile the single-step prediction for speed
    @jax.jit
    def predict_step(seq, params):
        z_dyn_seq = model.apply({'params': params}, seq, method=lambda m, x: m.dynamics(x))
        return z_dyn_seq[:, -1:, :]

    for _ in range(steps_to_generate):
        # Predict the next state from the current history
        # The Dynamics module returns the evolved state for each step.
        # We want the evolution of the LAST step in the current sequence.
        next_z = predict_step(current_z_seq, params)
        
        # Enforce manifold constraint (just in case, though dynamics should do it)
        norm = jnp.linalg.norm(next_z, axis=-1, keepdims=True)
        next_z = next_z / jnp.maximum(norm, epsilon)
        
        # Append to history
        current_z_seq = jnp.concatenate([current_z_seq, next_z], axis=1)
        
    # Extract the generated segment (excluding the history we started with)
    # The generated part starts at index gap_start
    generated_segment = current_z_seq[0, gap_start:]
    
    # Splice it back into the full latent sequence
    # We replace from gap_start to fill_end
    z_imputed_full = z_full_latent.at[gap_start:fill_end].set(generated_segment)
    
    # Decode the imputed sequence.
    imputed_output = model.apply({'params': params}, z_imputed_full, method=lambda module, x: module.vae.decode(x))
    return imputed_output

# --- Visualization ---

def plot_phase_portrait(z_traj, filename="phase_portrait.png"):
    """
    Plots the phase portrait of the latent trajectory using PCA.
    """
    # z_traj shape: (Seq_Len, Latent_Dim)
    # Center the data
    z_centered = z_traj - jnp.mean(z_traj, axis=0)
    
    # PCA via SVD
    # U: (T, T), S: (K,), Vt: (K, D) where K = min(T, D)
    _, _, Vt = jnp.linalg.svd(z_centered, full_matrices=False)
    
    # Project to top 2 components
    # z_pca = z_centered @ Vt.T[:, :2]
    z_pca = jnp.dot(z_centered, Vt[:2, :].T)
    
    plt.figure(figsize=(8, 8))
    plt.plot(z_pca[:, 0], z_pca[:, 1], label='Latent Trajectory', alpha=0.8)
    plt.scatter(z_pca[0, 0], z_pca[0, 1], color='green', marker='o', s=100, label='Start')
    plt.scatter(z_pca[-1, 0], z_pca[-1, 1], color='red', marker='x', s=100, label='End')
    
    # Draw direction arrows
    for i in range(0, len(z_pca)-1, len(z_pca)//10):
        plt.arrow(z_pca[i, 0], z_pca[i, 1], 
                  z_pca[i+1, 0] - z_pca[i, 0], 
                  z_pca[i+1, 1] - z_pca[i, 1], 
                  shape='full', lw=0, length_includes_head=True, head_width=0.05, color='black')

    plt.title("Latent Phase Portrait (PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved '{filename}'")

# --- Main ---

def main():
    key = jrandom.PRNGKey(0)
    
    seq_len = 256
    channels = 16
    latent_dim = 32
    num_epochs = 20
    batch_size = 64
    
    model = ManifoldFormer(input_dim=channels, latent_dim=latent_dim)
    
    synth_ds = SyntheticChordsDataset(num_samples=10 * batch_size, seq_len=seq_len, channels=channels, batch_size=batch_size)
    
    @jax.jit
    def train_step(state, x, y):
        grads = jax.grad(loss_fn)(state.params, state, x, y)
        state = state.apply_gradients(grads=grads)
        return state

    # Adjust sequence length for next-step prediction
    train_seq_len = seq_len - 1

    params = model.init(key, jnp.ones((batch_size, train_seq_len, channels)))['params']
    optimizer = optax.adamw(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(len(synth_ds)):
            x, y = synth_ds[i]
            state = train_step(state, x, y)
            loss = loss_fn(state.params, state, x, y)
            total_loss += loss
        avg_loss = total_loss / len(synth_ds)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
        
    print("\n--- Training Complete ---")
    
    # Reconstruct a full sequence for imputation demo
    sample_x, sample_y = synth_ds[0]
    sample_input = jnp.concatenate([sample_x[0], sample_y[0][-1][None, :]], axis=0)

    print("\n--- Demonstrating Geodesic Imputation ---")
    gap_start_idx = seq_len // 4
    gap_end_idx = seq_len // 2
    
    input_with_gap = np.copy(sample_input)
    input_with_gap[gap_start_idx:gap_end_idx, :] = 0.0
    
    imputed_output = geodesic_imputation(model, state.params, input_with_gap, gap_start_idx, gap_end_idx, key)
    
    save_audio(imputed_output, "imputed_audio_sample.wav")
    
    plt.figure(figsize=(15, 6))
    time_steps = np.arange(seq_len)
    
    plt.plot(time_steps, sample_input[:, 0], label='Original Signal (Channel 0)', alpha=0.7)
    plt.plot(time_steps, input_with_gap[:, 0], label='Masked Input (Channel 0)', alpha=0.7, linestyle='--')
    plt.plot(time_steps, imputed_output[:, 0], label='Imputed Signal (Channel 0)', color='red', linewidth=2)
    
    plt.axvspan(gap_start_idx, gap_end_idx - 1, color='gray', alpha=0.3, label='Imputed Region')
    
    plt.title('Geodesic Imputation Demonstration (Channel 0)')
    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("imputation_demo.png")
    plt.close()
    print("Saved 'imputation_demo.png'")

    print("\n--- Generating Phase Portrait ---")
    # Get latent trajectory for the sample input
    _, z_sample = model.apply({'params': state.params}, sample_input[None, ...], method=lambda m, x: m.vae(x))
    plot_phase_portrait(z_sample[0])

if __name__ == '__main__':
    main()