## Overview

This document specifies a **Riemannian variational autoencoder** (VAE) whose latent space is a **sphere** and whose temporal evolution is given by a **non‑Markovian ODE with power‑law memory** and Kuramoto‑style coupling, numerically integrated with a **projected RK4** scheme. I’ll derive and connect the main mathematical pieces.

---

## 1. Geometry of the Latent Space: Unit Hypersphere

### 1.1 Manifold and Tangent Space

The latent manifold is the unit sphere in \(\mathbb{R}^d\):

\[
\mathcal{M} = \mathbb{S}^{d-1} = \{ z \in \mathbb{R}^d : \|z\|_2 = 1\}.
\]

The sphere inherits the Riemannian metric from the ambient Euclidean space:
\[
g_z(u,v) = \langle u, v \rangle_{\mathbb{R}^d}, \quad u,v \in T_z\mathbb{S}^{d-1}.
\]

The **tangent space** at a point \(\mu \in \mathbb{S}^{d-1}\) is
\[
T_\mu \mathbb{S}^{d-1} = \{ v \in \mathbb{R}^d : \langle \mu, v \rangle = 0\}.
\]

This comes from the constraint \(\|z\|^2 = 1\). Let \(F(z) = \|z\|^2 - 1 = 0\) define the manifold. Then
\[
\nabla F(\mu) = 2\mu
\]
is normal to the manifold, so the tangent space is the orthogonal complement of \(\mu\).

### 1.2 Geodesics and the Exponential Map

On \(\mathbb{S}^{d-1}\) with the induced metric, **geodesics** are great circles. A geodesic starting at \(\mu\) with initial tangent velocity \(v\in T_\mu \mathbb{S}^{d-1}\) (nonzero) has the form
\[
\gamma(t) = \cos(\|v\| t)\,\mu + \sin(\|v\| t)\,\frac{v}{\|v\|}.
\]

This solves the geodesic equation with the constraint \(\|\gamma(t)\|=1\) and \(\gamma(0)=\mu\), \(\dot\gamma(0)=v\).

The **Riemannian exponential map** at \(\mu\) is defined by
\[
\exp_\mu(v) = \gamma(1) = \cos(\|v\|)\,\mu + \sin(\|v\|)\,\frac{v}{\|v\|},
\]
matching the given formula. For \(v=0\), continuity gives \(\exp_\mu(0) = \mu\).

### 1.3 Logarithmic Map on the Sphere

Given two points \(\mu, z \in \mathbb{S}^{d-1}\), the geodesic distance is
\[
d(\mu,z) = \arccos(\langle \mu, z \rangle).
\]

The **logarithmic map** \(\log_\mu(z) \in T_\mu \mathbb{S}^{d-1}\) is the inverse of \(\exp_\mu\) along the minimal geodesic (for \(z\) not antipodal to \(\mu\)). Let
\[
\theta = d(\mu,z) = \arccos(\langle \mu, z \rangle).
\]
The component of \(z\) orthogonal to \(\mu\) is
\[
w = z - \langle \mu, z \rangle \mu.
\]
If \(z = \exp_\mu(v)\) with \(\|v\| = \theta\), then
\[
z = \cos(\theta)\mu + \sin(\theta)\frac{v}{\theta}.
\]
Projecting onto the tangent direction:
\[
w = z - \langle\mu,z\rangle\mu = \sin(\theta)\frac{v}{\theta}.
\]
So
\[
v = \frac{\theta}{\sin(\theta)}\,w.
\]
Using \(\sin(\theta) = \sqrt{1 - \cos^2(\theta)} = \sqrt{1 - \langle \mu, z\rangle^2}\), we get
\[
\log_\mu(z) = \frac{\arccos(\langle \mu, z \rangle)}{\sqrt{1 - \langle \mu, z \rangle^2}}\, (z - \langle \mu, z \rangle \mu),
\]
again matching the given expression.

### 1.4 Projection onto Tangent Space

For any \(z \in \mathbb{S}^{d-1}\), the orthogonal projection of a vector \(v \in \mathbb{R}^d\) onto \(T_z\mathbb{S}^{d-1}\) is
\[
\Pi_{T_z\mathcal{M}}(v) = v - \langle v, z \rangle z.
\]
This subtracts the component of \(v\) along the normal direction \(z\).

### 1.5 Retraction

A **retraction** is a first-order approximation of the exponential map satisfying:  
(i) \(\mathcal{R}_z(0) = z\),  
(ii) \(\mathrm{D}\mathcal{R}_z(0) = \mathrm{id}_{T_z\mathcal{M}}\).

The given retraction
\[
\mathcal{R}_z(v) = \frac{z+v}{\|z+v\|}
\]
is just normalization back to the sphere. For small \(v\), \(\mathcal{R}_z(v)\) lies on the same great circle as \(\exp_z(v)\) to first order, hence is a valid retraction. It is cheaper than computing \(\exp_z(v)\), and is widely used in Riemannian optimization.

---

## 2. Stage I: Riemannian VAE Encoder

### 2.1 Tangent-Space Gaussian and Riemannian Reparameterization

A standard VAE uses a Gaussian in Euclidean latent space. Here, the latent space is \(\mathbb{S}^{d-1}\), so a natural choice is:

1. Choose a base point \(\mu \in \mathbb{S}^{d-1}\).
2. Place a Gaussian in the tangent space \(T_\mu \mathbb{S}^{d-1}\).
3. Map samples to the manifold via the exponential map:
   \[
   z = \exp_\mu(v).
   \]

This construction yields a **Riemannian normal distribution** (sometimes called a wrapped normal or exponential map Gaussian) on the manifold.

The encoder network outputs:

- A raw vector \(\mu_{\text{raw}} = f_\theta^\mu(X)\in\mathbb{R}^d\), normalized to
  \[
  \mu = \frac{\mu_{\text{raw}}}{\|\mu_{\text{raw}}\|}.
  \]
- Log-variance (or log-scale) parameters \(\sigma \in \mathbb{R}^d\) for a diagonal covariance in \(\mathbb{R}^d\).

### 2.2 Sampling in Tangent Space

The sampling procedure:

1. **Euclidean noise:**
   \[
   \epsilon \sim \mathcal{N}(0, I_d).
   \]

2. **Scale by variance:**
   \[
   v_{\text{raw}} = \epsilon \odot \exp(\sigma/2),
   \]
   so \(v_{\text{raw}} \sim \mathcal{N}(0, \Sigma)\) with diagonal \(\Sigma = \mathrm{diag}(\exp(\sigma))\).

3. **Project to tangent space:**
   \[
   v_{\text{tan}} = v_{\text{raw}} - \langle v_{\text{raw}}, \mu \rangle \mu.
   \]
   This enforces \(\langle v_{\text{tan}}, \mu \rangle = 0\), so \(v_{\text{tan}} \in T_\mu\mathbb{S}^{d-1}\).

4. **Map to manifold via exponential map:**
   \[
   z_0 = \exp_\mu(v_{\text{tan}}) = \cos(\|v_{\text{tan}}\|)\mu + \sin(\|v_{\text{tan}}\|)\frac{v_{\text{tan}}}{\|v_{\text{tan}}\|}.
   \]

This is a **Riemannian reparameterization trick**: sampling is expressed as a differentiable function of Gaussian noise and parameters \(\mu,\sigma\), enabling gradient-based training.

---

## 3. Stage II: Non‑Markovian Manifold ODE

The latent trajectory \(z(t)\) evolves on \(\mathbb{S}^{d-1}\) according to an ODE with memory and coupling.

### 3.1 ODE on a Manifold via Tangent Projection

We want a vector field \(f(z,t)\) such that \(z(t)\in\mathbb{S}^{d-1}\) for all \(t\). In Euclidean coordinates, we can define a raw force \(\tilde f(z,t)\) and then project onto the tangent space:

\[
\frac{dz}{dt} = \Pi_{T_{z(t)}\mathcal{M}}\big(\tilde f(z(t),t)\big).
\]

Given \(\|z\|=1\), if \(\frac{dz}{dt}\) is always orthogonal to \(z\), then:
\[
\frac{d}{dt}\|z\|^2 = 2\langle z, \dot z\rangle = 0,
\]
so the norm remains 1, at least in the continuous setting.

Here,
\[
\tilde f(z(t),t) = W_{\text{style}}\mathbf{M}(t) + F_{\text{couple}}(z(t)),
\]
and the actual ODE is
\[
\frac{dz}{dt} = \Pi_{T_{z(t)}\mathcal{M}} \left( W_{\text{style}}\mathbf{M}(t) + F_{\text{couple}}(z(t)) \right).
\]

### 3.2 Power-Law Memory: Fractional Integral

The memory term is
\[
\mathbf{M}(t) = \frac{1}{\Gamma(1-\gamma)} \int_0^t \frac{z(\tau)}{(t-\tau)^\gamma} \, d\tau,
\quad 0<\gamma<1.
\]

This is a **Riemann–Liouville fractional integral** of order \(1-\gamma\):
\[
(I^{1-\gamma} z)(t) = \frac{1}{\Gamma(1-\gamma)} \int_0^t (t-\tau)^{-\gamma} z(\tau)\,d\tau.
\]

Key points:

- The kernel \((t-\tau)^{-\gamma}\) decays as a power law, giving **long-range memory**.
- The \(\Gamma(1-\gamma)\) factor normalizes the operator to be consistent with fractional calculus definitions.
- In discrete time with step \(\Delta t\), this becomes a **causal convolution**
  \[
  \mathbf{M}[n] \approx \sum_{k=0}^{n-1} w_{n-k} z[k],
  \]
  with weights \(w_m \propto m^{-\gamma}\).

#### 3.2.1 Discrete Approximation and Code

The code:

```python
t = torch.arange(1, kernel_size + 1).float()
gamma_val = torch.exp(torch.lgamma(1 - torch.tensor(gamma)))
weights = (1.0 / gamma_val) * (t ** (-gamma))
```

implements the discrete kernel
\[
w_m = \frac{1}{\Gamma(1-\gamma)} m^{-\gamma}, \quad m = 1,\dots, \text{kernel\_size}.
\]

Flipping the kernel before convolution ensures **causality**: at time \(n\), the convolution uses \(z[n-m]\) for \(m\ge 1\), i.e., past values only.

This is a standard approximation of the continuous fractional integral in numerical fractional calculus.

### 3.3 Kuramoto-Type Coupling via Attention

The coupling term is
\[
F_{\text{couple}}(z(t)) = W_{\text{out}} \sum_{j=1}^{T_{\text{seq}}} \sin(\langle \hat{q}, \hat{k}_j \rangle)\, \hat{k}_j,
\]
with
\[
\hat{q} = \frac{W_Q z(t)}{\|W_Q z(t)\|},\quad
\hat{k}_j = \frac{W_K z_j}{\|W_K z_j\|}.
\]

This is inspired by the **Kuramoto model** of coupled oscillators. In its classical form, for phases \(\theta_i\),

\[
\dot{\theta}_i = \omega_i + \frac{K}{N}\sum_j \sin(\theta_j - \theta_i).
\]

Here:

- \(\hat{q}\) plays the role of a “query phase vector” of the current state.
- Each \(\hat{k}_j\) is a “key phase vector” of another time step (or token).
- The scalar
  \[
  \langle \hat{q}, \hat{k}_j \rangle \in [-1,1]
  \]
  measures alignment (analogous to phase difference). Taking \(\sin(\cdot)\) yields a **restoring force**: near alignment, small deviations produce approximately linear coupling; for larger misalignments, the sinusoidal nonlinearity appears.
- The force is a sum of these sinusoidal weights times directions \(\hat{k}_j\), then linearly transformed by \(W_{\text{out}}\).

This is structurally analogous to **self-attention**, but with:

- Scalar compatibility: \(\langle \hat{q}, \hat{k}_j \rangle\).
- Nonlinearity: \(\sin(\cdot)\) rather than softmax.
- Vector output: linear combination of keys, similar to attention over value vectors.

The subsequent projection \(\Pi_{T_{z(t)}\mathcal{M}}\) ensures the net coupling force is tangent to the sphere.

---

## 4. Stage II: Geometric Numerical Integration (Projected RK4)

To numerically solve
\[
\dot z = f(z,t), \quad z(t)\in\mathbb{S}^{d-1},
\]
we use a Runge–Kutta 4 scheme in the ambient \(\mathbb{R}^d\), but ensure each intermediate and final point is retracted back to the manifold.

### 4.1 Classical RK4 in Euclidean Space

For an ODE \(\dot y = f(y,t)\):

- \(k_1 = f(y_t, t)\)
- \(k_2 = f(y_t + \frac{h}{2}k_1, t + \frac{h}{2})\)
- \(k_3 = f(y_t + \frac{h}{2}k_2, t + \frac{h}{2})\)
- \(k_4 = f(y_t + h k_3, t + h)\)
- \(y_{t+h} = y_t + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)\).

This is fourth-order accurate in \(h\).

### 4.2 Projected RK4 on the Sphere

Here, \(f(z,t) = \Pi_{T_z\mathcal{M}}(\mathbf{v}_{\text{mem}} + F(z))\), where \(\mathbf{v}_{\text{mem}}\) is treated as constant over the step (frozen memory).

The scheme:

1. Compute memory drift \(\mathbf{v}_{\text{mem}}\) at time \(t\).

2. Slopes:
   - \(k_1 = \Pi_{T_{z_t}}(\mathbf{v}_{\text{mem}} + F(z_t))\).
   - Move halfway along \(k_1\) using the retraction:
     \[
     z_{k1} = \mathcal{R}_{z_t}\left(\frac{h}{2}k_1\right) = \frac{z_t + \frac{h}{2}k_1}{\|z_t + \frac{h}{2}k_1\|}.
     \]
   - \(k_2 = \Pi_{T_{z_{k1}}}(\mathbf{v}_{\text{mem}} + F(z_{k1}))\).
   - \(z_{k2} = \mathcal{R}_{z_t}\left(\frac{h}{2}k_2\right)\).
   - \(k_3 = \Pi_{T_{z_{k2}}}(\mathbf{v}_{\text{mem}} + F(z_{k2}))\).
   - \(z_{k3} = \mathcal{R}_{z_t}(h k_3)\).
   - \(k_4 = \Pi_{T_{z_{k3}}}(\mathbf{v}_{\text{mem}} + F(z_{k3}))\).

3. Combine slopes:
   \[
   v_{\text{final}} = \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
   \]

4. Retract to obtain next state:
   \[
   z_{t+1} = \mathcal{R}_{z_t}(v_{\text{final}}) = \frac{z_t + v_{\text{final}}}{\|z_t + v_{\text{final}}\|}.
   \]

This is an example of a **geometric integrator** for manifold ODEs:

- Intermediate points are always on \(\mathbb{S}^{d-1}\) due to retraction.
- Tangent projections guarantee the velocity is tangent.
- The method preserves the manifold constraint up to numerical roundoff, even for long trajectories.

In the context of a **music manifold**, this means that the latent representation of musical sequences evolves along smooth paths on a curved surface, with memory and coupling respecting the geometry.

---

## 5. Stage III: Decoder

The decoder maps the latent trajectory \(Z_{\text{out}}\) back to the signal space via a 1D transposed convolution:

\[
\hat{X} = \mathrm{ConvTranspose1d}(Z_{\text{out}}).
\]

Mathematically, this is a linear operator (plus nonlinearity if present) that performs:

- Upsampling in time (if stride > 1).
- Linear mixing of channels.

No additional geometry is imposed here; the important point is that the decoder is differentiable, allowing gradients from reconstruction loss to flow back through the ODE solver and the Riemannian VAE.

---

## 6. Optimization Objectives and Their Geometry

The training loss:
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Recon}} + \lambda_1 \mathcal{L}_{\text{Iso}} + \lambda_2 \mathcal{L}_{\text{GeoSmooth}} + \lambda_3 \mathcal{L}_{\text{Align}}.
\]

### 6.1 Reconstruction Loss

\[
\mathcal{L}_{\text{Recon}} = \|\hat{X} - X\|_F^2 = \sum_{c,t} (\hat X_{c,t} - X_{c,t})^2.
\]

This is the usual squared error; in a probabilistic view, it corresponds to a Gaussian likelihood with identity covariance.

### 6.2 Scaled Isometry Loss

\[
\mathcal{L}_{\text{Iso}} = \sum_{i,j} \left( s \cdot \arccos(\langle z_i, z_j \rangle) - \|x_i - x_j\|_2 \right)^2.
\]

Interpretation:

- \(\arccos(\langle z_i, z_j \rangle)\) is the **geodesic distance** on \(\mathbb{S}^{d-1}\) between \(z_i\) and \(z_j\).
- \(\|x_i - x_j\|_2\) is the Euclidean distance in input space.
- The scalar \(s>0\) is a learned scale to match the typical magnitude of distances.

This is a **manifold isometry regularizer**: it encourages the mapping \(x \mapsto z\) to be approximately distance-preserving (up to scale), i.e., an almost-isometric embedding of the data into the sphere.

In the “music manifold” context, similar musical patterns should be close along geodesics in latent space.

### 6.3 Geodesic Smoothness Loss

\[
\mathcal{L}_{\text{GeoSmooth}} = \sum_t \left( \arccos(\langle z_t, z_{t+1} \rangle) \right)^2.
\]

Since \(\arccos(\langle z_t, z_{t+1} \rangle)\) is the geodesic distance between consecutive latent states, this penalizes large geodesic jumps, enforcing **smooth trajectories** on the manifold.

For small angles \(\theta\), \(\arccos(\cos\theta) \approx \theta\), and
\[
\theta^2 \approx \|z_{t+1} - z_t\|^2/2,
\]
so this is akin to an \(L^2\) penalty on the discrete derivative of the trajectory, but using intrinsic distance instead of Euclidean norm.

This is especially relevant to temporal coherence in music: smooth evolution of latent “phase” or “style” variables.

### 6.4 Alignment Loss (Contrastive)

\[
\mathcal{L}_{\text{Align}} = -\sum_i \log \frac{\exp(\text{sim}(z_i^{(1)}, z_i^{(2)})/\tau)}{\sum_j \exp(\text{sim}(z_i^{(1)}, z_j^{(2)})/\tau)},
\]

where \(\text{sim}(\cdot,\cdot)\) is a similarity measure (often cosine similarity) and \(\tau>0\) is a temperature.

This is a standard **InfoNCE / contrastive** objective:

- Pairs \((z_i^{(1)}, z_i^{(2)})\) from different domains or augmentations but representing the same underlying content are treated as positives.
- Other pairs are negatives.
- The loss encourages **alignment across manifolds** or domains, so that equivalent musical content has similar latent representations even if played by different instruments, performers, or recording conditions.

---

## 7. Mathematical Basis of the Fractional Kernel Code

The function `get_fractional_kernel` implements the weights for the discrete approximation of the fractional integral kernel:

\[
k(t) = \frac{1}{\Gamma(1-\gamma)} t^{-\gamma}, \quad t>0.
\]

- `torch.lgamma(1 - gamma)` computes \(\log\Gamma(1-\gamma)\).
- `torch.exp(...)` exponentiates to get \(\Gamma(1-\gamma)\).
- `t ** (-gamma)` yields \(t^{-\gamma}\).
- Multiplying gives \(k(t) = \frac{1}{\Gamma(1-\gamma)} t^{-\gamma}\).

Flipping (`weights.flip(0)`) and reshaping for `conv1d` yields a causal convolution kernel approximating
\[
\mathbf{M}(t_n) \approx \sum_{m=1}^{K} \frac{1}{\Gamma(1-\gamma)} (m\Delta t)^{-\gamma} z(t_{n-m}).
\]

This is a discrete **Volterra convolution** with a power-law kernel, a standard tool in numerical fractional calculus and memory modeling.

---

## 8. Conceptual Integration: A Geometric, Fractional-Order Music Manifold

Putting everything together:

- The latent space is a **Riemannian manifold** (sphere) where similarity is measured by geodesic distance, not Euclidean distance.
- The encoder implements a **Riemannian VAE**, sampling from a Gaussian in tangent space and mapping via the exponential map.
- The temporal evolution is governed by a **fractional-order, non-Markovian ODE** with:
  - Long-range memory given by a **power-law fractional integral**.
  - **Kuramoto-style coupling** implemented as sinusoidal attention over other latent states.
- The ODE is integrated with a **retraction-based RK4** scheme to respect the manifold constraint.
- The loss terms enforce:
  - Faithful reconstruction of the signal.
  - Approximate isometry between original data space and manifold.
  - Smooth geodesic trajectories.
  - Cross-domain alignment when desired.

In the context of a **music manifold**, this formalism captures musical sequences as trajectories on a curved geometric space with memory and coupling, where distances and dynamics are intrinsically defined by the manifold structure and fractional temporal dependencies.