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
        self.sigma_head = nn.Dense(self.latent_dim)
        
        # Decoder: MLP to map from manifold back to signal space
        self.decoder_hidden = nn.Dense(128)
        self.decoder_out = nn.Dense(self.input_dim)

    def __call__(self, x, rng):
        # Encoder
        h = nn.relu(self.encoder_conv1(x))
        h = nn.relu(self.encoder_conv2(h))
        
        # Predict mu (on manifold) and sigma (tangent space concentration)
        mu_raw = self.encoder_dense(h)
        mu = mu_raw / jnp.maximum(jnp.linalg.norm(mu_raw, axis=-1, keepdims=True), 1e-8)
        
        # Sigma parameterizes the scale in tangent space
        sigma = self.sigma_head(h)
        sigma = nn.softplus(sigma) + 1e-5 # Positive scale
        
        # Tangent Space Sampling (Riemannian VAE)
        # 1. Sample epsilon ~ N(0, I)
        epsilon = jrandom.normal(rng, shape=mu.shape)
        
        # 2. Project to Tangent Space of mu
        # v_raw = epsilon * sigma
        v_raw = epsilon * sigma
        dot = jnp.sum(v_raw * mu, axis=-1, keepdims=True)
        v_tan = v_raw - dot * mu
        
        # 3. Exponential Map: z = exp_mu(v_tan)
        norm_v = jnp.linalg.norm(v_tan, axis=-1, keepdims=True)
        z = jnp.cos(norm_v) * mu + jnp.sin(norm_v) * (v_tan / jnp.maximum(norm_v, 1e-8))
        
        # Decoder (Pointwise)
        recon = self.decode(z)
        
        return recon, z, mu, sigma

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
            padding='CAUSAL',
            kernel_init=nn.initializers.zeros # Fix: Start with zero memory force
        )(z)

        # 2. Prepare Kuramoto Components (Keys/Values)
        # We treat the input sequence 'z' as the background field for the attention.
        # Ideally, for RK4, the field should be dynamic, but for efficiency in parallel training,
        # we assume the 'Keys' and 'Values' are fixed by the input trajectory,
        # and only the 'Query' (the particle being evolved) changes during the RK4 substeps.
        
        # "Pick a lock with your own key" -> Use intrinsic geometry (Key = Self)
        # Remove projections to match the sphere geometry directly.
        # k_proj_layer = nn.Dense(self.latent_dim, name="K_proj")
        # q_proj_layer = nn.Dense(self.latent_dim, name="Q_proj")
        out_proj_layer = nn.Dense(self.latent_dim, name="Out_proj", kernel_init=nn.initializers.zeros) # Fix: Start with zero coupling force
        
        # Sakaguchi-Kuramoto Phase Lag
        # A learnable phase shift that prevents amplitude death and promotes limit cycles.
        phase_lag = self.param('phase_lag', nn.initializers.zeros, (1,))

        # k = k_proj_layer(z)
        # k_hat = k / jnp.maximum(jnp.linalg.norm(k, axis=-1, keepdims=True), 1e-8)
        k_hat = z # z is already on manifold

        # Causal Mask for Attention
        # mask[i, j] = 1 if i >= j else 0
        mask = jnp.tril(jnp.ones((T, T)))
        mask = mask[None, :, :] # (1, T, T)

        # Define the Vector Field Function for RK4
        # z_curr: (B, T, D) - The current estimate of the state at each time step
        def vector_field(z_curr):
            # Project z_curr to Query
            # q = q_proj_layer(z_curr)
            # q_hat = q / jnp.maximum(jnp.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
            q_hat = z_curr # Intrinsic query
            
            # Causal Attention: Query (Current) vs Keys (History/Context)
            dots = jnp.einsum('btd,bTd->btT', q_hat, k_hat)
            
            # Apply Mask: Set future positions to -inf before softmax? 
            # But this is Kuramoto (sin), not Softmax. 
            # We just zero out the contributions from the future.
            # Sakaguchi-Kuramoto: sin(theta_j - theta_i + alpha)
            attn_weights = jnp.sin(dots + phase_lag) * mask
            
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
        # Pillar 3: Simplify Geometry - Remove Dynamic Rotation
        # self.align = DynamicCayleyRotation(self.latent_dim)
        self.dynamics = Dynamics(self.latent_dim)
        self.iso_scale = self.param('iso_scale', nn.initializers.ones, (1,))

    def __call__(self, x, rng):
        recon_vae, z, mu, sigma = self.vae(x, rng)
        
        # Pillar 3: Trust the Encoder (No explicit alignment)
        # z_aligned, R = self.align(z)
        # z_dyn = self.dynamics(z_aligned)
        
        z_dyn = self.dynamics(z)
        
        # Decode the dynamic state
        out = self.vae.decode(z_dyn)
        
        return out, recon_vae, z, z_dyn, self.iso_scale, mu, sigma

# --- Loss Function ---

# --- Loss Function ---

# --- Loss Function ---

def fractional_fidelity_loss(pred, target, gammas, kernel_size=128):
    """
    Computes fidelity loss by convolving with a bank of fractional kernels.
    This measures alignment across different 'temporal horizons' defined by gamma.
    Uses Log-Magnitude Spectral Loss to prevent energy squashing.
    """
    loss = 0.0
    B, T, C = pred.shape
    
    for gamma in gammas:
        raw_kernel = get_fractional_kernel(gamma, kernel_size) # (K, 1, 1)
        
        # FFT convolution
        fft_len = T + kernel_size - 1
        
        # Pad kernel
        kernel_padded = jnp.pad(raw_kernel.squeeze(), (0, fft_len - kernel_size))
        kernel_fft = jnp.fft.rfft(kernel_padded)
        
        # Pad signals
        pred_T = pred.transpose(0, 2, 1) # (B, C, T)
        target_T = target.transpose(0, 2, 1)
        
        pred_padded = jnp.pad(pred_T, ((0,0), (0,0), (0, kernel_size - 1)))
        target_padded = jnp.pad(target_T, ((0,0), (0,0), (0, kernel_size - 1)))
        
        pred_fft = jnp.fft.rfft(pred_padded, axis=-1)
        target_fft = jnp.fft.rfft(target_padded, axis=-1)
        
        # Convolve in freq domain
        pred_conv_fft = pred_fft * kernel_fft[None, None, :]
        target_conv_fft = target_fft * kernel_fft[None, None, :]
        
        # Log-Magnitude Spectral Loss
        # L = || log(|F(y)|) - log(|F(hat_y)|) ||
        # Add epsilon to prevent log(0)
        eps = 1e-8
        mag_pred = jnp.abs(pred_conv_fft) + eps
        mag_target = jnp.abs(target_conv_fft) + eps
        
        log_mag_pred = jnp.log(mag_pred)
        log_mag_target = jnp.log(mag_target)
        
        l_mag = jnp.mean(jnp.abs(log_mag_pred - log_mag_target))
        
        # Phase Consistency Loss
        # 1 - cos(theta_pred - theta_target)
        # We weight this less than magnitude to prioritize energy presence first
        phase_pred = jnp.angle(pred_conv_fft)
        phase_target = jnp.angle(target_conv_fft)
        l_phase = jnp.mean(1.0 - jnp.cos(phase_pred - phase_target))
        
        loss += l_mag + 1.0 * l_phase
        
    return loss / len(gammas)




def loss_fn(params, state, x, y_pred_target, y_recon_target, oracle_z, rng, return_mse=False):
    pred, recon_vae, z, z_dyn, iso_scale, mu, sigma = state.apply_fn({'params': params}, x, rng)
    
    # Track MSE for reference (not used in backprop)
    l_recon = jnp.mean((pred - y_pred_target) ** 2)
    l_recon_vae = jnp.mean((recon_vae - y_recon_target) ** 2)
    mse_loss = l_recon + l_recon_vae
    
    # --- Geometric Supervision (Tracking Only) ---
    # Pillar 2: Only Use Spectral Loss. We track Oracle Distance but do not optimize it.
    l_oracle_enc = jnp.mean((z - oracle_z) ** 2)
    
    # Geodesic smoothness on the sphere
    dot_product = jnp.sum(z_dyn * z, axis=-1).clip(-1.0 + 1e-6, 1.0 - 1e-6)
    geodesic_dist = jnp.arccos(dot_product)
    l_smooth = jnp.mean(geodesic_dist ** 2)
    
    # Scaled Isometry Loss
    dist_x = jnp.sqrt(jnp.sum((x[:, :, None, :] - x[:, None, :, :]) ** 2, axis=-1) + 1e-8)
    dot_z = jnp.einsum('btd,bTd->btT', z, z).clip(-1.0 + 1e-6, 1.0 - 1e-6)
    dist_z = jnp.arccos(dot_z)
    l_iso = jnp.mean((iso_scale * dist_z - dist_x) ** 2)
    
    # KL Divergence
    l_kl = -0.5 * jnp.mean(1 + jnp.log(sigma ** 2) - sigma ** 2)
    
    # --- Fractional Fidelity Loss ---
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
    l_frac = fractional_fidelity_loss(pred, y_pred_target, gammas, kernel_size=128)
    
    # Latent Resonant Entropy (Power-based)
    z_fft = jnp.fft.rfft(z_dyn, axis=1)
    z_power = jnp.abs(z_fft) ** 2 
    z_prob = z_power / (jnp.sum(z_power, axis=1, keepdims=True) + 1e-8)
    l_entropy = -jnp.mean(jnp.sum(z_prob * jnp.log(z_prob + 1e-8), axis=1))
    
    # Base loss (geometric terms only, for tracking)
    base_loss = 0.1 * l_smooth + 0.01 * l_iso + l_oracle_enc
    
    # TOTAL LOSS
    # Pillar 2: Only Spectral Loss + Entropy
    spectral_loss = 1.0 * l_frac + 0.05 * l_entropy
    
    if return_mse:
        return spectral_loss, mse_loss, base_loss
    return spectral_loss

# --- Oracle & Validation ---

def compute_oracle_trajectory(seq_len, freqs, phase_shifts, latent_dim):
    """
    Computes the 'Perfect' latent trajectory based on the ground truth oscillators.
    Embeds the analytic signal (sin, cos) of each frequency onto the hypersphere.
    """
    t = np.linspace(0, 4 * np.pi, seq_len)
    
    # We construct a high-dimensional state from the oscillators
    # For each freq, we have sin and cos components (Analytic Signal)
    components = []
    for f in freqs:
        # We average the phase shifts across channels to get the "fundamental" phase for this freq
        # Or better, we just take the base phase. 
        # In the dataset, phase_shift depends on channel. 
        # The latent space should capture the "global" dynamics.
        # Let's assume the latent space captures the fundamental oscillators.
        components.append(np.sin(f * t))
        components.append(np.cos(f * t))
        
    # Stack components
    oracle_raw = np.stack(components, axis=-1) # (T, 2*Num_Freqs)
    
    # Pad or project to latent_dim
    T, D_raw = oracle_raw.shape
    if D_raw < latent_dim:
        # Pad with noise or zeros? Zeros is safer for "perfect" geometry.
        padding = np.zeros((T, latent_dim - D_raw))
        oracle_raw = np.concatenate([oracle_raw, padding], axis=-1)
    elif D_raw > latent_dim:
        # PCA projection? Or just truncate?
        oracle_raw = oracle_raw[:, :latent_dim]
        
    # Normalize to Sphere (The Manifold Constraint)
    norm = np.linalg.norm(oracle_raw, axis=-1, keepdims=True)
    oracle_z = oracle_raw / (norm + 1e-8)
    
    return oracle_z

def procrustes_distance(z_learned, z_oracle):
    """
    Computes the Procrustes distance between learned and oracle trajectories.
    Finds optimal rotation R to minimize ||z_learned - z_oracle @ R||.
    Returns the RMSE after alignment.
    """
    # z_learned: (T, D)
    # z_oracle: (T, D)
    
    # Center data (though they are on sphere, centering helps align "clouds")
    # UPDATE: Do NOT center. We want to align the spheres themselves, which are already centered at 0.
    # Centering destroys the spherical constraint.
    z_learned_c = z_learned
    z_oracle_c = z_oracle
    
    # SVD for optimal rotation
    # M = A.T @ B
    M = z_learned_c.T @ z_oracle_c
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    # Align
    z_aligned = z_learned_c @ R
    
    # RMSE
    mse = np.mean((z_aligned - z_oracle_c) ** 2)
    return np.sqrt(mse)

# --- Data Generation ---

class SyntheticChordsDataset:
    def __init__(self, num_samples=1000, seq_len=128, channels=64, batch_size=16, latent_dim=21):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        
    def __len__(self):
        return self.num_samples // self.batch_size
    
    def __getitem__(self, idx):
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        
        clean_signals = []
        noisy_signals = []
        oracle_trajectories = []
        
        # Make data deterministic per index to allow "grokking"
        np.random.seed(idx)
        
        for _ in range(self.batch_size):
            freqs = np.random.choice([1, 2, 3, 5, 8], size=3, replace=False)
            clean_signal = np.zeros((self.seq_len, self.channels))
            noisy_signal = np.zeros((self.seq_len, self.channels))
            
            # Generate Oracle for this sample
            # We use the freqs selected.
            oracle_z = compute_oracle_trajectory(self.seq_len, freqs, None, self.latent_dim)
            oracle_trajectories.append(oracle_z)
            
            for ch in range(self.channels):
                phase_shift = (ch / self.channels) * 2 * np.pi
                wave = (np.sin(freqs[0] * t + phase_shift) + 
                        0.5 * np.sin(freqs[1] * t + phase_shift) + 
                        0.25 * np.sin(freqs[2] * t + phase_shift))
                clean_signal[:, ch] = wave
                noisy_signal[:, ch] = wave + np.random.normal(0, 0.1, size=len(t))
            
            clean_signals.append(clean_signal)
            noisy_signals.append(noisy_signal)
            
        clean_signals = np.array(clean_signals)
        noisy_signals = np.array(noisy_signals)
        oracle_trajectories = np.array(oracle_trajectories)
        
        # Input: Noisy [0:-1]
        # Target: Clean [1:]
        # Oracle: [0:-1] (aligned with input state)
        
        return noisy_signals[:, :-1, :], clean_signals[:, 1:, :], clean_signals[:, :-1, :], oracle_trajectories[:, :-1, :]

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
    # We need to pass a key for the VAE
    # Ensure input is batched (1, T, C)
    sequence_batched = sequence_with_gap[None, ...]
    _, z_full_latent, _, _ = model.apply({'params': params}, sequence_batched, key, method=lambda module, x, rng: module.vae(x, rng))
    
    # Pillar 3: Fix Imputation Consistency
    # We do NOT align here (Dynamic Rotation removed).
    # We pass the full history through dynamics to get the "Dynamics Output" distribution for the known parts.
    
    # Run dynamics on the full latent sequence (or at least up to gap)
    # This gives us z_dyn for the history part, which matches what the decoder expects.
    z_dyn_full = model.apply({'params': params}, z_full_latent, method=lambda m, x: m.dynamics(x))
    
    # We start with the valid history up to the gap
    # Note: z_dyn_full[t] corresponds to prediction for t+1 (if h=1).
    # But for autoregression, we need the STATE at the boundary.
    # The state at the boundary is z_full_latent[gap_start-1].
    # We want to evolve from there.
    
    # Wait, for the autoregressive loop, we need to feed inputs to dynamics.
    # The input to dynamics is "z" (latent state).
    # The output of dynamics is "z_next" (evolved state).
    # So we need to maintain a sequence of "z" values.
    
    # History z values: z_full_latent[:gap_start]
    current_z_seq = z_full_latent[:, :gap_start, :] # (1, T, D)
    
    # We need to generate up to the end of the shadow region
    receptive_field = 16
    fill_end = min(gap_end + receptive_field, original_seq_len)
    steps_to_generate = fill_end - gap_start
    
    print(f"Autoregressively generating {steps_to_generate} steps using learned dynamics...")
    
    # JIT compile the single-step prediction for speed
    @jax.jit
    def predict_step(seq, params):
        # Dynamics returns the evolved state for the sequence
        z_dyn_seq = model.apply({'params': params}, seq, method=lambda m, x: m.dynamics(x))
        # The last element is the prediction for the next step
        return z_dyn_seq[:, -1:, :]

    generated_zs = []
    
    for _ in range(steps_to_generate):
        # Predict the next state from the current history
        next_z = predict_step(current_z_seq, params)
        
        # Enforce manifold constraint
        norm = jnp.linalg.norm(next_z, axis=-1, keepdims=True)
        next_z = next_z / jnp.maximum(norm, epsilon)
        
        # Append to history for next step
        current_z_seq = jnp.concatenate([current_z_seq, next_z], axis=1)
        generated_zs.append(next_z)
        
    # Now we construct the sequence to DECODE.
    # The decoder expects "z_dyn" (outputs of dynamics).
    # For the history part (0 to gap_start), we use z_dyn_full[:gap_start].
    # For the gap part, we use the generated z values (which ARE outputs of dynamics from the previous step).
    
    # generated_zs is a list of (1, 1, D) tensors.
    generated_segment = jnp.concatenate(generated_zs, axis=1) # (1, Steps, D)
    
    # Splicing:
    # History: z_dyn_full[0 : gap_start] (This is the dynamics output for the history inputs)
    # Gap: generated_segment (This is the dynamics output for the gap inputs)
    # Tail: z_dyn_full[fill_end:] (This is the dynamics output for the tail inputs - though these might be corrupted by the gap in the input)
    
    z_imputed_dyn = z_dyn_full.at[:, gap_start:fill_end, :].set(generated_segment)
    
    # Decode the dynamic state
    imputed_output = model.apply({'params': params}, z_imputed_dyn, method=lambda module, x: module.vae.decode(x))
    
    return imputed_output[0] # Return unbatched

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
    latent_dim = 21
    num_epochs = 5
    batch_size = 64
    
    model = ManifoldFormer(input_dim=channels, latent_dim=latent_dim)
    
    synth_ds = SyntheticChordsDataset(num_samples=10 * batch_size, seq_len=seq_len, channels=channels, batch_size=batch_size, latent_dim=latent_dim)
    
    @jax.jit
    def train_step(state, x, y_pred, y_recon, oracle_z, rng):
        rng, key = jrandom.split(rng)
        grads = jax.grad(loss_fn)(state.params, state, x, y_pred, y_recon, oracle_z, key)
        state = state.apply_gradients(grads=grads)
        return state, rng

    # Adjust sequence length for next-step prediction
    train_seq_len = seq_len - 1

    # Init with RNG
    init_rng, train_rng = jrandom.split(key)
    params = model.init(init_rng, jnp.ones((batch_size, train_seq_len, channels)), init_rng)['params']
    optimizer = optax.adamw(1e-3)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    
    print("Starting training (JIT compilation will occur on first batch, this may take 30-60s)...")
    for epoch in range(num_epochs):
        total_spectral = 0
        total_mse = 0
        total_base = 0
        total_oracle_dist = 0
        
        for i in range(len(synth_ds)):
            x, y_pred, y_recon, oracle_z = synth_ds[i]
            state, train_rng = train_step(state, x, y_pred, y_recon, oracle_z, train_rng)
            
            # For logging, we need a key for loss_fn too if we call it outside
            train_rng, log_key = jrandom.split(train_rng)
            spectral, mse, base = loss_fn(state.params, state, x, y_pred, y_recon, oracle_z, log_key, return_mse=True)
            total_spectral += spectral
            total_mse += mse
            total_base += base
            
            # Compute Oracle Distance (Validation)
            # Get current latent state
            _, _, z, _, _, _, _ = state.apply_fn({'params': state.params}, x, log_key)
            
            # Compute Procrustes for the batch
            # Average over batch
            batch_dist = 0
            for b in range(batch_size):
                batch_dist += procrustes_distance(z[b], oracle_z[b])
            total_oracle_dist += batch_dist / batch_size
            
        avg_spectral = total_spectral / len(synth_ds)
        avg_mse = total_mse / len(synth_ds)
        avg_base = total_base / len(synth_ds)
        avg_oracle = total_oracle_dist / len(synth_ds)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Spectral: {avg_spectral:.4f}, Oracle Dist: {avg_oracle:.4f}, MSE: {avg_mse:.4f}")
        
    print("\n--- Training Complete ---")
    
    # Reconstruct a full sequence for imputation demo
    sample_x, sample_y_pred, sample_y_recon, _ = synth_ds[0]
    sample_input = jnp.concatenate([sample_x[0], sample_y_pred[0][-1][None, :]], axis=0)

    print("\n--- Demonstrating Geodesic Imputation ---")
    gap_start_idx = seq_len // 4
    gap_end_idx = seq_len // 2
    
    input_with_gap = np.copy(sample_input)
    input_with_gap[gap_start_idx:gap_end_idx, :] = 0.0
    
    # Pass a key for the VAE sampling in imputation
    impute_key, _ = jrandom.split(key)
    imputed_output = geodesic_imputation(model, state.params, input_with_gap, gap_start_idx, gap_end_idx, impute_key)
    
    save_audio(imputed_output, "outputs/beta_imputed_audio_sample.wav")
    
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
    plt.savefig("outputs/beta_imputation_demo.png")
    plt.close()
    print("Saved 'beta_imputation_demo.png'")

    print("\n--- Generating Phase Portrait ---")
    # Get latent trajectory for the sample input
    # Use the mean (mu) for the phase portrait to show the "clean" manifold
    _, _, z_sample_mu, _ = model.apply({'params': state.params}, sample_input[None, ...], key, method=lambda m, x, rng: m.vae(x, rng))
    plot_phase_portrait(z_sample_mu[0], filename="outputs/beta_phase_portrait.png")

if __name__ == '__main__':
    main()