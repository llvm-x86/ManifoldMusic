import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax.training import train_state
import flax.linen as nn
import numpy as np
import optax
import sys

# --- Patch A/B: The Trivial Baseline ---

class EuclideanEncoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        # Simple MLP or Conv Encoder
        # Input: (Batch, Time, Channels)
        # Output: (Batch, Time, Latent)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x

class SphericalEncoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        
        # Project to Sphere
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        z = x / jnp.maximum(norm, 1e-8)
        return z

class IdentityDynamics(nn.Module):
    @nn.compact
    def __call__(self, z):
        return z

class SimpleDynamics(nn.Module):
    latent_dim: int
    
    @nn.compact
    def __call__(self, z):
        # Simple Residual: z_next = z + f(z)
        delta = nn.Dense(self.latent_dim)(z)
        z_new = z + delta
        
        # Retract to sphere
        norm = jnp.linalg.norm(z_new, axis=-1, keepdims=True)
        z_new = z_new / jnp.maximum(norm, 1e-8)
        return z_new

def get_fractional_kernel(gamma, kernel_size):
    t = jnp.arange(1, kernel_size + 1, dtype=jnp.float32)
    gamma_val = jnp.exp(jax.lax.lgamma(1 - jnp.array(gamma)))
    weights = (1.0 / gamma_val) * (t ** (-gamma))
    return weights[::-1].reshape(kernel_size, 1, 1)

class FullDynamics(nn.Module):
    latent_dim: int
    memory_gamma: float = 0.5
    memory_kernel_size: int = 64

    @nn.compact
    def __call__(self, z):
        B, T, D = z.shape
        
        # Memory
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
            kernel_init=nn.initializers.zeros # Initialize to zero force
        )(z)
        
        # Kuramoto
        out_proj_layer = nn.Dense(self.latent_dim, name="Out_proj", kernel_init=nn.initializers.zeros) # Initialize to zero force
        phase_lag = self.param('phase_lag', nn.initializers.zeros, (1,))
        
        k_hat = z
        mask = jnp.tril(jnp.ones((T, T)))[None, :, :]
        
        def vector_field(z_curr):
            q_hat = z_curr
            dots = jnp.einsum('btd,bTd->btT', q_hat, k_hat)
            attn_weights = jnp.sin(dots + phase_lag) * mask
            kuramoto_force_raw = jnp.einsum('btT,bTd->btd', attn_weights, k_hat)
            kuramoto_force = out_proj_layer(kuramoto_force_raw)
            
            total_force = memory_force + kuramoto_force
            
            dot_prod = jnp.sum(total_force * z_curr, axis=-1, keepdims=True)
            tangent_vel = total_force - dot_prod * z_curr
            return tangent_vel
            
        # RK4
        h = 1.0
        def retract(z_base, v):
            z_new = z_base + v
            return z_new / jnp.maximum(jnp.linalg.norm(z_new, axis=-1, keepdims=True), 1e-8)
            
        k1 = vector_field(z)
        z_k2 = retract(z, 0.5 * h * k1)
        k2 = vector_field(z_k2)
        z_k3 = retract(z, 0.5 * h * k2)
        k3 = vector_field(z_k3)
        z_k4 = retract(z, h * k3)
        k4 = vector_field(z_k4)
        
        v_final = (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        z_next = retract(z, v_final)
        return z_next

class EuclideanDecoder(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, z):
        # Simple MLP Decoder
        x = nn.Dense(64)(z)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class PatchModel(nn.Module):
    input_dim: int
    latent_dim: int
    patch_level: str = 'A' # 'A', 'B', 'C', 'D', 'E'

    def setup(self):
        if self.patch_level in ['C', 'D']:
            self.encoder = SphericalEncoder(self.latent_dim)
        else:
            self.encoder = EuclideanEncoder(self.latent_dim)
            
        if self.patch_level == 'D':
            self.dynamics = SimpleDynamics(self.latent_dim)
        elif self.patch_level == 'E':
            self.dynamics = FullDynamics(self.latent_dim)
        else:
            self.dynamics = IdentityDynamics()
            
        self.decoder = EuclideanDecoder(self.input_dim)

    def __call__(self, x):
        z = self.encoder(x)
        z_dyn = self.dynamics(z)
        recon = self.decoder(z_dyn)
        return recon, z

# --- Data Generation (Copied from beta.py but simplified) ---

class SyntheticChordsDataset:
    def __init__(self, num_samples=1000, seq_len=128, channels=16, batch_size=64):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.batch_size = batch_size
        
    def __len__(self):
        return self.num_samples // self.batch_size
    
    def __getitem__(self, idx):
        t = np.linspace(0, 4 * np.pi, self.seq_len)
        batch_x = []
        batch_y = []
        
        np.random.seed(idx)
        
        for _ in range(self.batch_size):
            freqs = np.random.choice([1, 2, 3, 5, 8], size=3, replace=False)
            signal = np.zeros((self.seq_len, self.channels))
            for ch in range(self.channels):
                phase = (ch / self.channels) * 2 * np.pi
                wave = (np.sin(freqs[0] * t + phase) + 
                        0.5 * np.sin(freqs[1] * t + phase))
                signal[:, ch] = wave
            
            # Input: [0:-1], Target: [1:] (Next step prediction)
            # Or for Identity task: Input [0:], Target [0:]
            # Patch A says "Map x[t] to x[t] (Identity) or x[t+1] (Next Step)"
            # Let's do Identity first as it's easiest.
            batch_x.append(signal)
            batch_y.append(signal) 
            
        return np.array(batch_x), np.array(batch_y)

# --- Loss Functions ---

def mse_loss_fn(params, state, x, y):
    recon, _ = state.apply_fn({'params': params}, x)
    loss = jnp.mean((recon - y) ** 2)
    return loss

def spectral_loss_fn(params, state, x, y):
    recon, _ = state.apply_fn({'params': params}, x)
    
    # Simple FFT loss
    recon_fft = jnp.fft.rfft(recon, axis=1)
    y_fft = jnp.fft.rfft(y, axis=1)
    
    # Log-Magnitude Spectral Loss
    mag_loss = jnp.mean(jnp.abs(jnp.log(jnp.abs(recon_fft) + 1e-8) - jnp.log(jnp.abs(y_fft) + 1e-8)))
    
    # Phase Consistency Loss
    phase_pred = jnp.angle(recon_fft)
    phase_target = jnp.angle(y_fft)
    l_phase = jnp.mean(1.0 - jnp.cos(phase_pred - phase_target))
    
    return mag_loss + l_phase

# --- Training Loop ---

def run_patch(patch_level):
    print(f"\n--- Running Patch {patch_level} ---")
    
    # Config
    latent_dim = 16
    channels = 16
    seq_len = 64
    batch_size = 64
    epochs = 5
    lr = 1e-3
    
    # Init
    model = PatchModel(input_dim=channels, latent_dim=latent_dim, patch_level=patch_level)
    key = jrandom.PRNGKey(0)
    dummy_input = jnp.ones((batch_size, seq_len, channels))
    params = model.init(key, dummy_input)['params']
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    
    ds = SyntheticChordsDataset(num_samples=1000, seq_len=seq_len, channels=channels, batch_size=batch_size)
    
    # Loss selection
    if patch_level == 'A':
        loss_fn = mse_loss_fn
        print("Objective: MSE Loss")
    elif patch_level == 'B':
        loss_fn = spectral_loss_fn
        print("Objective: Spectral Loss (Log-Mag + Phase)")
    elif patch_level in ['C', 'D', 'E']:
        def combined_loss_fn(params, state, x, y):
            return mse_loss_fn(params, state, x, y) + spectral_loss_fn(params, state, x, y)
        loss_fn = combined_loss_fn
        print("Objective: Spectral + MSE (Spherical Constraint)")
    else:
        raise ValueError("Unknown patch level")
        
    @jax.jit
    def train_step(state, x, y):
        grads = jax.grad(loss_fn)(state.params, state, x, y)
        state = state.apply_gradients(grads=grads)
        loss = loss_fn(state.params, state, x, y)
        
        metrics = {'loss': loss}
        if patch_level in ['C', 'D', 'E']:
            l_mse = mse_loss_fn(state.params, state, x, y)
            l_spec = spectral_loss_fn(state.params, state, x, y)
            metrics['mse'] = l_mse
            metrics['spec'] = l_spec
            
        return state, metrics

    for epoch in range(epochs):
        total_metrics = {}
        count = 0
        for i in range(len(ds)):
            x, y = ds[i]
            state, metrics = train_step(state, x, y)
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            count += 1
            
        log_str = f"Epoch {epoch+1}: Loss = {total_metrics['loss']/count:.4f}"
        if 'mse' in total_metrics:
            log_str += f" (MSE: {total_metrics['mse']/count:.4f}, Spec: {total_metrics['spec']/count:.4f})"
        print(log_str)

if __name__ == "__main__":
    # run_patch('A')
    # run_patch('B')
    # run_patch('C')
    # run_patch('D')
    run_patch('E')
