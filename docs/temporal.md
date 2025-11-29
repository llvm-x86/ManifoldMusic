# Formal Specification: The Geometric Power-Law ManifoldFormer

This document provides a consolidated, mathematically rigorous specification for the ManifoldFormer system, merging the core concepts of a Riemannian VAE with a memory-augmented manifold ODE.

---

## 1. Global Geometric Assumption

The system assumes latent dynamics evolve on a compact Riemannian manifold $\mathcal{M}$, specifically the unit Hypersphere $\mathbb{S}^{d-1}$ embedded i   n $\mathbb{R}^d$.

*   **Manifold Definition:** $\mathcal{M} = \{ z \in \mathbb{R}^d \mid \|z\|_2 = 1 \}$
*   **Tangent Space ($T_\mu\mathcal{M}$):** The vector space tangent to $\mathcal{M}$ at point $\mu$. For the sphere, $T_\mu\mathcal{M} = \{ v \in \mathbb{R}^d \mid \langle \mu, v \rangle = 0 \}$.
*   **Exponential Map ($\exp_\mu$):** Maps a tangent vector $v \in T_\mu\mathcal{M}$ back to the manifold following a geodesic.
    $$\exp_\mu(v) = \cos(\|v\|_2)\mu + \sin(\|v\|_2)\frac{v}{\|v\|_2}$$ 
*   **Logarithmic Map ($\log_\mu$):** The inverse of the exponential map, which maps a manifold point $z$ to the tangent space at $\mu$.
    $$\log_\mu(z) = \frac{\arccos(\langle \mu, z \rangle)}{\sqrt{1 - \langle \mu, z \rangle^2}} (z - \langle \mu, z \rangle \mu)$$ 
*   **Retraction Operator ($\mathcal{R}_z$):** A computationally simpler way to project a tangent vector back to the manifold, used in the ODE solver.
    $$\mathcal{R}_z(v) = \frac{z+v}{\|z+v\|_2}$$ 

---

## 2. Stage I: Riemannian VAE (The Encoder)

To ensure the variational posterior respects the manifold geometry, we employ a **Tangent Space Gaussian** distribution wrapped via the exponential map.

### A. Feature Extraction

Given an input sequence $X \in \mathbb{R}^{C \times T}$, convolutional networks parameterize the distribution:

*   A base point on the manifold: $\mu_{raw} = f_\theta^\mu(X); \quad \mu = \mu_{raw} / \|\mu_{raw}\|_2$
*   Concentration parameters for the tangent space: $\sigma = f_\theta^\sigma(X) \in \mathbb{R}^d$

### B. Tangent Space Sampling

We sample noise in Euclidean space, project it onto the tangent space of $\mu$, scale it, and then map the result to the manifold.

1.  **Euclidean Noise:** $\epsilon \sim \mathcal{N}(0, I_d)$
2.  **Tangent Projection:** Scale the noise and project it to ensure orthogonality to $\mu$.
    $$v_{raw} = \epsilon \odot \exp(\sigma/2)$$
    $$v_{tan} = v_{raw} - \langle v_{raw}, \mu \rangle \mu$$ 
3.  **Manifold Mapping:** The initial latent state $z_0$ is obtained by applying the exponential map.
    $$z_0 = \exp_\mu(v_{tan})$$ 

---

## 3. Stage II: Dynamics (Memory-Augmented Manifold ODE)

We model temporal evolution using a **Memory-Augmented ODE** driven by a Power-Law memory kernel to simulate non-Markovian dynamics.

### A. The Differential Equation

The dynamics are governed by a first-order ODE on the manifold's tangent space.

$$\frac{dz}{dt} = \Pi_{T_{z(t)}\mathcal{M}} 

 \left( \underbrace{W_{style} \cdot \mathbf{M}(t)}_{\text{Memory Force}} + \underbrace{F_{couple}(z(t))}_{\text{Kuramoto Coupling}} \right)$$ 

*   **Tangent Projection Operator:** $\Pi_{T_z\mathcal{M}}(v) = v - \langle v, z \rangle z$

*   **1. Power-Law Memory Integral ($\mathbf{M}(t)$):** Captures long-range dependencies with Gamma-normalized decay.
    $$\mathbf{M}(t) = \frac{1}{\Gamma(1-\gamma)} \int_0^t \frac{z(\tau)}{(t-\tau)^\gamma} d\tau$$ 
    This is implemented in discrete time as a causal convolution with kernel $k[n] \propto n^{-\gamma}$.

*   **2. Projected Kuramoto Coupling ($F_{couple}$):** Standard attention-based phase coupling, projected onto the tangent space.
    $$F_{couple}(z) = W_{out} \sum_{j=1}^{T_{seq}} \sin(\langle \hat{q}, \hat{k}_j \rangle) \cdot \hat{k}_j$$ 
    Where $\hat{q} = W_Q z(t) / \|W_Q z(t)\|_2$ and $\hat{k}_j = W_K z_j / \|W_K z_j\|_2$ are projected query and key vectors.

### B. The Geometric Solver (Projected RK4)

To maintain the manifold constraint $z(t) \in \mathbb{S}^{d-1}$ during integration, we use a **Retraction-based Runge-Kutta 4** method. For a single step $h$ from $z_t$ to $z_{t+1}$:

1.  **Drift Pre-calculation:** Compute the memory vector $\mathbf{v}_{mem}$ (assumed constant over the step $h$). 
2.  **Slopes:**
    *   $k_1 = \Pi_{T_{z_t}}(\mathbf{v}_{mem} + F(z_t))$
    *   $z_{k1} = \mathcal{R}_{z_t}(\frac{h}{2}k_1)$
    *   $k_2 = \Pi_{T_{z_{k1}}}(\mathbf{v}_{mem} + F(z_{k1}))$
    *   $z_{k2} = \mathcal{R}_{z_t}(\frac{h}{2}k_2)$
    *   $k_3 = \Pi_{T_{z_{k2}}}(\mathbf{v}_{mem} + F(z_{k2}))$
    *   $z_{k3} = \mathcal{R}_{z_t}(h k_3)$
    *   $k_4 = \Pi_{T_{z_{k3}}}(\mathbf{v}_{mem} + F(z_{k3}))$
3.  **Update:**
    $$v_{final} = \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$
    $$z_{t+1} = \mathcal{R}_{z_t}(v_{final})$$ 

---

## 4. Stage III: The Decoder

The final latent trajectory $Z_{out}$ is mapped back to the signal space $\hat{X}$ using a 1D Transposed Convolutional network.

$$\hat{X} = \text{ConvTranspose1d}(Z_{out})$$ 

---

## 5. Optimization Objectives

The model is trained end-to-end by minimizing a composite loss function that balances reconstruction with geometric and dynamic regularization.

$$\mathcal{L}_{total} = \mathcal{L}_{Recon} + \lambda_1 \mathcal{L}_{Iso} + \lambda_2 \mathcal{L}_{GeoSmooth} + \lambda_3 \mathcal{L}_{Align}$$ 

| Term | Equation | Purpose |
| :--- | :--- | :--- |
| **Reconstruction** | $\| \hat{X} - X \|_F^2$ | Ensures data fidelity (Frobenius norm). |
| **Scaled Isometry** | $\sum_{i,j} \left( s \cdot \arccos(\langle z_i, z_j \rangle) - \|x_i - x_j\|_2 \right)^2$ | Enforces that geodesic distances on the sphere (with learnable scale $s$) match Euclidean distances in input space. |
| **Geodesic Smoothness**| $\sum_{t} \left( \arccos(\langle z_t, z_{t+1} \rangle) \right)^2$ | Penalizes high-frequency jitter using the intrinsic manifold distance (arc length) to encourage phase locking. |
| **Alignment (Optional)**| $-\sum_{i} \log \frac{\exp(\text{sim}(z_i^{(1)}, z_i^{(2)})/\tau)}{\sum_j \exp(\text{sim}(z_i^{(1)}, z_j^{(2)})/\tau)}$ | Optional contrastive loss to align manifolds across different subjects or domains. |

---

## 6. Implementation Notes (Python/PyTorch)

### Gamma-Normalized Kernel

To ensure the mathematical validity of the discrete fractional integral approximation, the convolutional kernel can be generated as follows:

```python
import torch
import torch.nn.functional as F

def get_fractional_kernel(gamma, kernel_size, device):
    """
    Generates the discrete approximation of the fractional integral kernel
    (t-tau)^(-gamma) / Gamma(1-gamma).
    """
    t = torch.arange(1, kernel_size + 1, device=device).float()
    
    # For rigor, include the Gamma function normalization, though this
    # is often absorbed into a learnable scale in deep learning models.
    gamma_val = torch.exp(torch.lgamma(1 - torch.tensor(gamma)))
    weights = (1.0 / gamma_val) * (t ** (-gamma))
    
    # Reshape for conv1d: (Out_channels, In_channels, Time)
    return weights.flip(0).view(1, 1, -1)
```

