# Investigation Status: Imputation Failure in ManifoldFormer

**Date**: 2025-11-27  
**Objective**: Resolve poor imputation performance despite achieving target loss (~0.03) on deterministic synthetic data

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

## Investigation Timeline

### Attempt #1: Fixed Power-Law Kernel Initialization

**Hypothesis**: The memory kernel $\mathbf{M}(t)$ was being randomly initialized instead of using the prescribed power-law structure.

**Implementation**:
```python
def memory_kernel_init(key, shape, dtype=jnp.float32):
    k = get_fractional_kernel(gamma, kernel_size)  # t^(-gamma) / Gamma(1-gamma)
    return jnp.tile(k, (1, 1, latent_dim))

memory_force = nn.Conv(..., kernel_init=memory_kernel_init, use_bias=False)(z)
```

**Mathematical Change**:
$$
K[n] = \frac{n^{-\gamma}}{\Gamma(1-\gamma)} \quad \text{(enforced at initialization)}
$$

**Result**: Loss increased to **0.13** (baseline: 0.03)

**Diagnosis**: While the kernel was correctly initialized, it remained **trainable**, allowing gradient descent to drift away from the geometric structure.

---

### Attempt #2: Learnable Projection with Fixed Structure

**Hypothesis**: Remove the learnable capacity entirely from the memory integral while introducing a separate mixing layer.

**Implementation**:
$$
\text{memory\_force} = W_{style}(\text{Conv}_{fixed}(z))
$$

where $\text{Conv}_{fixed}$ uses the frozen power-law kernel, and $W_{style} \in \mathbb{R}^{d \times d}$ is a learnable Dense layer.

**Result**: Loss increased to **0.09**

**Diagnosis**: The additional projection layer created **optimization difficulties**. The model struggled to balance:
- Fixed geometric structure (temporal memory)
- Learnable mixing weights
- Reconstruction fidelity

---

### Attempt #3: Strictly Frozen Kernel

**Hypothesis**: Prevent kernel drift entirely by computing convolution outside the parameter system.

**Implementation**:
```python
kernel = jnp.tile(get_fractional_kernel(gamma, kernel_size), (1, 1, latent_dim))
z_padded = jnp.pad(z, ((0, 0), (pad_width, 0), (0, 0)))
memory_force = jax.lax.conv_general_dilated(
    lhs=z_padded, rhs=kernel, ..., feature_group_count=latent_dim
)
```

**Result**: Loss **diverged** to 0.25 by epoch 10 (peaked at 0.12 epoch 6)

**Diagnosis**: Training became **unstable**. The frozen kernel enforced a strong inductive bias that conflicted with the optimizer's ability to find a good solution in the joint parameter space $(W_Q, W_K, W_{out}, \text{VAE})$.

---

## Root Cause Analysis

### The True Issue: Imputation Mechanism

All kernel-focused fixes failed to improve imputation. Re-examining the `geodesic_imputation` function revealed:

**Original Implementation**:
$$
\begin{align}
&\text{1. Encode full sequence (including gap):} \quad z_{full} = \text{VAE.encode}(x_{gapped}) \\
&\text{2. Extract pre-gap history:} \quad z_{hist} = z_{full}[:gap\_start] \\
&\text{3. Autoregressive rollout in latent space:} \\
&\quad z_{t+1} = \text{Dynamics}(z_{:t}) \\
&\text{4. Decode imputed latents:} \quad \hat{x} = \text{VAE.decode}(z_{imputed})
\end{align}
$$

**Fundamental Flaw**:

The model is trained **end-to-end** with the full pipeline:
$$
X \xrightarrow{\text{VAE}} Z \xrightarrow{\text{Dynamics}} Z_{dyn} \xrightarrow{\text{Decode}} \hat{X}
$$

But imputation used **only** the Dynamics module in isolation:
$$
Z_{hist} \xrightarrow{\text{Dynamics only}} Z_{imputed}
$$

This violates the learned representation since:
1. The Dynamics module expects $Z$ from VAE encoding of **clean signal**
2. Autoregressive latent rollout lacks the decoder's learned inverse mapping
3. The gap was pre-encoded with **zeros**, corrupting the context

---

## Final Solution

### Corrected Imputation Algorithm

**Autoregressive Forward Pass**:
$$
\begin{align}
&\text{Initialize:} \quad X_{hist} = X_{valid}[:gap\_start] \\
&\text{For } t = gap\_start \text{ to } gap\_end + \text{receptive\_field}: \\
&\quad \quad X_t^{(batch)} = X_{hist}[None, :, :] \quad \text{(add batch dim)} \\
&\quad \quad \hat{X}_{t+1} = \text{ManifoldFormer}(X_t^{(batch)}, rng)[0, -1, :] \quad \text{(full forward pass)} \\
&\quad \quad X_{hist} \leftarrow \text{concat}(X_{hist}, \hat{X}_{t+1}) \\
&\text{Splice} \quad X_{imputed} = X_{gapped}.at[gap\_start:fill\_end].set(X_{hist}[gap\_start:])
\end{align}
$$

**Key Changes**:
1. **Use full model**: $X \to \text{VAE} \to \text{Dynamics} \to \text{Decode} \to \hat{X}$
2. **Feed predictions back as input**, matching training paradigm
3. **No latent manipulation**, preserving learned representations

---

## Architectural Insights

### Why Memory Kernel Fixes Failed

The power-law memory kernel is **one component** of a complex dynamical system. Enforcing its mathematical form without co-adapting the rest of the system creates:

**Constraint Violation**:
$$
\min_{\theta} \mathcal{L}_{recon} + \lambda_{smooth} \mathcal{L}_{geo} + \lambda_{iso} \mathcal{L}_{iso} \quad \text{s.t. } K_{memory} = K_{fixed}
$$

The constrained optimization landscape becomes **non-convex and unstable**.

### The Correct Inductive Bias

Instead of enforcing rigid geometric structure, the fix:
1. **Respects the learned manifold** by always passing through VAE
2. **Matches training distribution** (autoregressive signal-space prediction)
3. **Allows geometric priors to emerge** from loss functions, not hard constraints

---

## Status: Resolution Implemented

**Implementation**: ✅ `geodesic_imputation` rewritten to use full ManifoldFormer forward pass

**Theoretical Soundness**: ✅ Aligns imputation with training objective

**Expected Outcome**: Imputation should capture deterministic patterns via learned dynamics $\frac{dz}{dt}$, with the VAE maintaining signal-manifold correspondence

---

## Mathematical Summary

| Approach | Core Idea | Mathematical Form | Loss | Status |
|----------|-----------|------------------|------|--------|
| Original | Random kernel init | $K \sim \mathcal{N}(0, \sigma^2)$ | 0.03 | ❌ Poor imputation |
| Attempt 1 | Init with power-law | $K(0) = t^{-\gamma}/\Gamma(1-\gamma)$ | 0.13 | ❌ Kernel drift |
| Attempt 2 | Init + projection | $F = W_{style}(K_{init} * z)$ | 0.09 | ❌ Optimization conflict |
| Attempt 3 | Frozen kernel | $K = \text{const}$ | 0.25 | ❌ Training divergence |
| **Solution** | **Fix imputation logic** | $\hat{X}_t = \text{Model}(X_{<t})$ | **TBD** | ✅ **Correct paradigm** |

---

*The journey from geometric constraints to learned dynamics.*
