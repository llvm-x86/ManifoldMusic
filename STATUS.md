# Investigation Status: Imputation Failure in ManifoldFormer

**Date**: 2025-11-27  
**Objective**: Resolve poor imputation performance despite achieving target loss (~0.03) on deterministic synthetic data

---

## Critical Lessons Learned (Do Not Repeat)

### 1. Imputation Strategy: Latent vs. Signal Space
**Mistake**: Performing imputation in **signal space** (Input $\to$ VAE $\to$ Dynamics $\to$ Decode $\to$ Input).
**Why it failed**: This feedback loop accumulates error from the VAE's encoder/decoder at every step. The signal degrades rapidly because the model is not a perfect identity map.
**Correction**: Perform autoregressive rollout entirely in **latent space** (Encode Context $\to$ Evolve Latent State $\to$ Decode Result).
**Mechanism**:
$$
z_{t+1} = \text{Dynamics}(z_t) \quad \text{(Latent Evolution)}
$$
Only decode $\hat{X}$ at the very end. This preserves the phase/frequency information encoded in the manifold state without corruption.

### 2. Dynamics Modeling: Discrete vs. Continuous
**Mistake**: Using multi-step RK4 (substeps=4) with small step size ($h=0.25$) for discrete sequence data.
**Why it failed**: The data is discrete (time steps). Forcing a continuous solver to take multiple substeps between data points complicates the optimization landscape and creates a mismatch between the model's "time" and the data's index.
**Correction**: Use **single-step RK4** ($h=1.0$).
**Mechanism**:
$$
z_{t+1} = \text{RK4}(z_t, h=1.0)
$$
This effectively treats the ODE solver as a sophisticated Recurrent Neural Network (RNN) cell that respects the manifold geometry.

### 3. Optimization: Learning Rate & JIT
**Mistake**: Using a low learning rate (`1e-4`) with the complex dynamics model.
**Correction**: Increase learning rate to `1e-3`. The manifold optimization landscape is well-conditioned enough for larger steps.
**Note**: JAX JIT compilation takes significant time (30-60s) on the first batch. **Always inform the user** via print statements to prevent "hang" diagnosis.

### 4. Architecture: ManifoldFormer
**Correct Architecture**:
- **Encoder**: Causal Conv $\to$ Dense $\to$ Tangent Space Projection $\to$ Exp Map
- **Dynamics**: Power-Law Memory (Long-term) + Kuramoto Attention (Short-term/Coupling)
- **Decoder**: Dense MLP (Pointwise)

---

## System Architecture

The ManifoldFormer implements a three-stage pipeline for temporal sequence modeling on a Riemannian manifold $\mathcal{M} = \mathbb{S}^{d-1}$:

### Stage I: Riemannian VAE

**Encoding to Tangent Space**:
$$
\begin{align}
\mu_{raw} &= f_\theta^\mu(X), \quad \mu = \frac{\mu_{raw}}{\|\mu_{raw}\|_2} \in \mathbb{S}^{d-1} \\
\sigma &= \text{softplus}(f_\theta^\sigma(X)) + \epsilon
\end{align}
$$

**Sampling via Exponential Map**:
$$
\begin{align}
\epsilon &\sim \mathcal{N}(0, I_d) \\
v_{tan} &= \text{Proj}_{T_\mu \mathcal{M}}(\epsilon \odot \sigma) = (\epsilon \odot \sigma) - \langle \epsilon \odot \sigma, \mu \rangle \mu \\
z_0 &= \exp_\mu(v_{tan}) = \cos(\|v_{tan}\|)\mu + \sin(\|v_{tan}\|)\frac{v_{tan}}{\|v_{tan}\|}
\end{align}
$$

### Stage II: Manifold ODE with Power-Law Memory

**Dynamics Equation**:
$$
\frac{dz}{dt} = \Pi_{T_z \mathcal{M}} \left( W_{style} \cdot \mathbf{M}(t) + F_{kuramoto}(z) \right)
$$

where:

- **Power-Law Memory Integral**:
$$
\mathbf{M}(t) = \frac{1}{\Gamma(1-\gamma)} \int_0^t \frac{z(\tau)}{(t-\tau)^\gamma} d\tau
$$

- **Kuramoto Coupling**:
$$
F_{kuramoto}(z) = W_{out} \sum_j \sin(\langle \hat{q}, \hat{k}_j \rangle) \cdot \hat{k}_j
$$

- **Tangent Projection**:
$$
\Pi_{T_z \mathcal{M}}(v) = v - \langle v, z \rangle z
$$

---

## Mathematical Summary of Fixes

| Component | Previous (Failed) | Current (Success) | Reason |
|-----------|-------------------|-------------------|--------|
| **Imputation** | Signal Space (AR) | **Latent Space (AR)** | Avoids VAE error accumulation |
| **Solver** | RK4 (4 substeps, h=0.25) | **RK4 (1 step, h=1.0)** | Matches discrete data structure |
| **Kernel** | Hard-coded / Frozen | **Learnable (Init as Power-Law)** | Allows data adaptation |
| **LR** | 1e-4 | **1e-3** | Better convergence |

---

*The journey from geometric constraints to learned dynamics.*
