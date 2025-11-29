Below is a structured derivation and explanation of the mathematical concepts in `manifold former.md`, with emphasis on the underlying geometry, probabilistic modeling, and dynamical systems. I’ll also occasionally connect to the “music manifold” idea at a conceptual level.

---

## 1. Riemannian Latent Embedding

### 1.1. Variational reparameterization in Euclidean space

The starting point is the standard reparameterization trick used in Variational Autoencoders (VAEs).

An encoder maps an input \(x\) (here: EEG, but conceptually also music features) to parameters \(\mu(x), \sigma(x)\) of a Gaussian distribution in latent space:
\[
q_\phi(z \mid x) = \mathcal{N}\big(z; \mu(x), \operatorname{diag}(\exp(\sigma(x)))\big).
\]

To sample \(z\) in a differentiable way, one writes:
\[
z = \mu + \epsilon \odot \exp(\sigma/2), \quad \epsilon \sim \mathcal{N}(0, I).
\]
This expresses the random variable \(z\) as a deterministic transformation of \(\epsilon\), allowing gradients to flow through \(\mu,\sigma\).

### 1.2. Moving from Euclidean to Riemannian manifolds

The ManifoldFormer adds a projection \(\Pi_{\mathcal{M}}\) to constrain samples to a manifold \(\mathcal{M}\), such as:

- The unit hypersphere \(\mathbb{S}^{d-1} = \{ z \in \mathbb{R}^d : \|z\|_2 = 1 \}\).
- The Poincaré ball model of hyperbolic space:
  \[
  \mathbb{B}^d = \{ z \in \mathbb{R}^d : \|z\|_2 < 1 \}.
  \]

The reparameterization becomes:
\[
z = \Pi_{\mathcal{M}}\big(\mu + \epsilon \odot \exp(\sigma/2)\big).
\]

This means: first sample a point in Euclidean space using the Gaussian reparameterization, then project that point onto (or into) the desired manifold.

#### 1.2.1. Projection onto the sphere \(\mathbb{S}^{d-1}\)

For the unit sphere, the natural projection is:
\[
\Pi_{\mathbb{S}^{d-1}}(y) = \frac{y}{\|y\|_2}.
\]
- This enforces \(\|z\|_2 = 1\).
- The Riemannian metric on \(\mathbb{S}^{d-1}\) is inherited from \(\mathbb{R}^d\), restricted to the tangent space at each point.

The geodesic distance between two points \(u, v \in \mathbb{S}^{d-1}\) is:
\[
d_{\mathbb{S}^{d-1}}(u, v) = \arccos(\langle u, v \rangle),
\]
where \(\langle \cdot,\cdot\rangle\) is the Euclidean inner product.

#### 1.2.2. Projection into the Poincaré ball \(\mathbb{B}^d\)

For the Poincaré ball of curvature \(-1\) (or some negative curvature \(-c\)), a simple “norm clipping” projection is:
\[
\Pi_{\mathbb{B}^d}(y) =
\begin{cases}
y, & \|y\|_2 < 1,\\
\frac{y}{\|y\|_2}(1 - \varepsilon), & \|y\|_2 \ge 1,
\end{cases}
\]
for a small \(\varepsilon>0\) to ensure strict inclusion.

The Riemannian metric in the Poincaré ball (curvature \(-1\)) is:
\[
g_z = \lambda_z^2 \, I, \quad \lambda_z = \frac{2}{1 - \|z\|_2^2},
\]
so the line element is:
\[
ds^2 = \lambda_z^2 \, d\|z\|_2^2.
\]

The geodesic distance between \(u, v \in \mathbb{B}^d\) is:
\[
d_{\mathbb{B}^d}(u, v)
= \operatorname{arcosh}\left(1 + 2\frac{\|u - v\|_2^2}{(1 - \|u\|_2^2)(1 - \|v\|_2^2)}\right).
\]

### 1.3. Why Riemannian manifolds for latent embeddings?

Conceptually, the idea is that the latent representations of EEG (or music) live on a curved manifold where:

- Hierarchical or tree-like structure is well captured by negative curvature (hyperbolic space).
- Angular relationships or direction-based features are well captured by spherical geometry.

The projection \(\Pi_{\mathcal{M}}\) enforces that the latent variable \(z\) respects the geometry chosen for \(\mathcal{M}\).

---

## 2. Geodesic-Aware Attention Mechanism

Standard Transformer attention is:
\[
\text{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V.
\]

Each row of \(QK^\top\) is a similarity score between a query and all keys, usually a dot product in Euclidean space.

### 2.1. Incorporating geodesic distances

ManifoldFormer modifies attention to:
\[
\text{Attention}(Q, K, V) =
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} - \lambda D_{geo}\right)V,
\]
where:

- \(D_{geo} \in \mathbb{R}^{n_q \times n_k}\) is a matrix of geodesic distances on \(\mathcal{M}\):
  \[
  (D_{geo})_{ij} = d_{\mathcal{M}}(z^{(Q)}_i, z^{(K)}_j).
  \]
- \(\lambda > 0\) is a scaling parameter controlling the strength of geometric regularization.

The term \(-\lambda D_{geo}\) penalizes pairs that are far apart on the manifold, making attention more local in the intrinsic geometry.

### 2.2. Relation to kernel methods

Think of the standard attention scores as:
\[
S_{ij}^{\text{base}} = \frac{\langle q_i, k_j\rangle}{\sqrt{d_k}}.
\]
Adding the geodesic term:
\[
S_{ij} = S_{ij}^{\text{base}} - \lambda d_{\mathcal{M}}(z^{(Q)}_i, z^{(K)}_j).
\]

Exponentiating inside softmax:
\[
\exp(S_{ij}) = \exp\left(S_{ij}^{\text{base}}\right) \cdot \exp\left(-\lambda d_{\mathcal{M}}(z^{(Q)}_i, z^{(K)}_j)\right).
\]

Thus, the effective attention kernel is:
\[
K_{\text{eff}}(i,j) =
\exp\left(S_{ij}^{\text{base}}\right) \cdot \exp\left(-\lambda d_{\mathcal{M}}(z^{(Q)}_i, z^{(K)}_j)\right).
\]

The factor \(\exp(-\lambda d_{\mathcal{M}})\) is reminiscent of a radial basis function kernel but defined on a Riemannian manifold, not just \(\mathbb{R}^d\). It enforces that points far in geodesic distance receive exponentially less attention.

For a music manifold: this encourages attention to focus on temporally or structurally nearby musical states in the intrinsic music-geometry, not just in raw feature space.

---

## 3. Manifold-Constrained Neural ODE

### 3.1. Neural ODE formulation

A Neural ODE describes continuous-time evolution of a hidden state \(z(t)\) via:
\[
\frac{dz(t)}{dt} = f_\theta(z(t), t, c(t)),
\]
where:

- \(z(t) \in \mathcal{M}\) is the latent state on the manifold.
- \(c(t)\) is a context function (e.g., Fourier time embeddings, state-space models, ResNet features).
- \(f_\theta\) is a neural network parameterized by \(\theta\).

In Euclidean space, this is straightforward. On a manifold, one must ensure that the vector field \(f_\theta(z(t), t, c(t))\) is tangent to \(\mathcal{M}\) at \(z(t)\), so that the ODE flow stays on \(\mathcal{M}\).

#### 3.1.1. Tangent constraint

For a Riemannian manifold \(\mathcal{M}\) embedded in \(\mathbb{R}^D\), the tangent space at \(z\) is \(T_z\mathcal{M}\). To ensure \(f_\theta(z) \in T_z\mathcal{M}\), one can:

- Define \(f_\theta\) in the tangent space using an orthonormal basis of \(T_z\mathcal{M}\).
- Or compute a Euclidean vector field \(\tilde{f}_\theta(z)\) and project it onto the tangent space:
  \[
  f_\theta(z) = P_{T_z\mathcal{M}}\big(\tilde{f}_\theta(z)\big).
  \]

For the sphere \(\mathbb{S}^{d-1}\), the tangent space at \(z\) is:
\[
T_z\mathbb{S}^{d-1} = \{ v \in \mathbb{R}^d : \langle v, z \rangle = 0 \}.
\]
Projection:
\[
P_{T_z\mathbb{S}^{d-1}}(v) = v - \langle v, z\rangle z.
\]

For the Poincaré ball, the tangent space is still \(\mathbb{R}^d\), but the Riemannian metric changes norms and inner products. The ODE solver should ideally use the Riemannian metric when computing norms and step sizes (Riemannian ODEs).

### 3.2. Numerical ODE solver (Dopri5)

The solver notation:
\[
\hat{z}_{t+1}^{ode} = ODESolver_{Dopris}(z_t, f_\theta)
\]
refers to a Dormand–Prince (Dopri5) solver, a 5th-order Runge–Kutta method with adaptive step size.

In discrete time, the solver approximates:
\[
\hat{z}_{t+1}^{ode} \approx z(t + \Delta t), \quad \text{given } z(t) = z_t.
\]

In practice, for manifold-constrained dynamics, one often:

1. Integrates in a local coordinate chart or tangent space.
2. Maps back to the manifold using an exponential map \(\exp_z\) or a projection \(\Pi_{\mathcal{M}}\).

For example, on a sphere:
- Integrate in \(T_{z_t}\mathbb{S}^{d-1}\) to get a tangent update \(v\).
- Use the exponential map:
  \[
  \exp_{z_t}(v) = \cos(\|v\|) z_t + \sin(\|v\|)\frac{v}{\|v\|}.
  \]

### 3.3. State transition with Transformer and LSTM history

The full state update is:
\[
z_{t+1} = \hat{z}_{t+1}^{ode} \oplus Tf_{auto}(z_{\le t}) \oplus LSTM_{bi}(z_{\le t}),
\]
where \(\oplus\) denotes concatenation.

Interpretation:

- \(\hat{z}_{t+1}^{ode}\): continuous-time prediction from the Neural ODE based on current state and context.
- \(Tf_{auto}(z_{\le t})\): Transformer-based autoregressive encoding of the history \(z_0,\dots,z_t\).
- \(LSTM_{bi}(z_{\le t})\): bidirectional LSTM encoding of the same history.

The concatenated vector may then be linearly projected back into the manifold latent space (e.g., via a fully connected layer followed by \(\Pi_{\mathcal{M}}\)) to ensure it lies in \(\mathcal{M}\).

Mathematically, one can think of:
\[
\tilde{z}_{t+1} = \phi\big(\hat{z}_{t+1}^{ode}, Tf_{auto}(z_{\le t}), LSTM_{bi}(z_{\le t})\big),
\]
with \(\phi\) a neural network, and then:
\[
z_{t+1} = \Pi_{\mathcal{M}}(\tilde{z}_{t+1}).
\]

For a music manifold, this describes how musical states evolve in continuous time (Neural ODE) while also attending to long-range structural dependencies (Transformer) and local sequence patterns (LSTM).

---

## 4. Geometric Learning Objectives

The total loss:
\[
\mathcal{L}_{total} = \mathcal{L}_{recon} + \alpha \mathcal{L}_{geo} + \beta \mathcal{L}_{align},
\]
combines reconstruction, geometry preservation, and cross-domain alignment.

### 4.1. Reconstruction loss \(\mathcal{L}_{recon}\)

In a VAE-like framework:
- Encoder: \(x \mapsto q_\phi(z \mid x)\).
- Decoder: \(z \mapsto p_\psi(x \mid z)\).

Typical reconstruction loss (for continuous data) is:
\[
\mathcal{L}_{recon} = -\mathbb{E}_{q_\phi(z \mid x)}[\log p_\psi(x \mid z)].
\]
For Gaussian decoder:
\[
p_\psi(x \mid z) = \mathcal{N}(x; \hat{x}(z), \sigma_x^2 I),
\]
then:
\[
\mathcal{L}_{recon} \propto \|x - \hat{x}(z)\|_2^2.
\]

For music, \(\mathcal{L}_{recon}\) might be cross-entropy (for symbolic sequences) or spectral loss (for audio spectrograms).

### 4.2. Geometric consistency loss \(\mathcal{L}_{geo}\)

Given inputs \(x_i\) with Euclidean distances \(\|x_i - x_j\|_2\), and embeddings \(z_i\) with manifold distances \(\|z_i - z_j\|_{\mathcal{M}} = d_{\mathcal{M}}(z_i, z_j)\), the geometric consistency loss is:
\[
\mathcal{L}_{geo} =
\sum_{i,j} \left| \|z_i - z_j\|_{\mathcal{M}} - \|x_i - x_j\|_2 \right|^2.
\]

This is essentially a multidimensional scaling (MDS)-style objective, but with:

- Source distances measured in Euclidean input space.
- Target distances measured in the manifold metric.

#### 4.2.1. Explicit forms of \(\|z_i - z_j\|_{\mathcal{M}}\)

- Sphere:
  \[
  \|z_i - z_j\|_{\mathcal{M}} = d_{\mathbb{S}^{d-1}}(z_i, z_j) = \arccos(\langle z_i, z_j\rangle).
  \]

- Poincaré ball:
  \[
  \|z_i - z_j\|_{\mathcal{M}} = d_{\mathbb{B}^d}(z_i, z_j)
  = \operatorname{arcosh}\left(1 + 2\frac{\|z_i - z_j\|_2^2}{(1 - \|z_i\|_2^2)(1 - \|z_j\|_2^2)}\right).
  \]

This loss encourages the mapping \(x \mapsto z\) to be approximately distance-preserving, at least locally (if the sum is restricted to neighbors).

For music: local neighborhoods might correspond to similar timbral textures, harmonic contexts, or rhythmic patterns. \(\mathcal{L}_{geo}\) then enforces that such local relationships are preserved in the music manifold.

### 4.3. Contrastive alignment loss \(\mathcal{L}_{align}\)

The alignment loss is:
\[
\mathcal{L}_{align}
= -\sum_i \log
\frac{\exp(\operatorname{sim}(z_i^{(1)}, z_i^{(2)})/\tau)}
{\sum_j \exp(\operatorname{sim}(z_i^{(1)}, z_j^{(2)})/\tau)},
\]
where:

- \(z_i^{(1)}\) and \(z_i^{(2)}\) are embeddings of two “views” of the same underlying example (e.g., two subjects, two modalities, or two augmentations).
- \(\operatorname{sim}(\cdot,\cdot)\) is a similarity measure, often cosine similarity:
  \[
  \operatorname{sim}(u, v) = \frac{\langle u, v\rangle}{\|u\|_2 \|v\|_2}.
  \]
- \(\tau > 0\) is a temperature parameter.

This is an InfoNCE-style contrastive loss. For each anchor \(z_i^{(1)}\):

- The positive sample is \(z_i^{(2)}\).
- The negatives are \(z_j^{(2)}\) for \(j \neq i\).

#### 4.3.1. Probabilistic interpretation

Define:
\[
p_i(j) = \frac{\exp(\operatorname{sim}(z_i^{(1)}, z_j^{(2)})/\tau)}
{\sum_k \exp(\operatorname{sim}(z_i^{(1)}, z_k^{(2)})/\tau)}.
\]

Then:
\[
\mathcal{L}_{align} = -\sum_i \log p_i(i).
\]

Minimizing \(\mathcal{L}_{align}\) maximizes the log-likelihood of correctly matching each example across domains, encouraging:

- High similarity between matched pairs \((z_i^{(1)}, z_i^{(2)})\).
- Low similarity between mismatched pairs \((z_i^{(1)}, z_j^{(2)}), j\neq i\).

In a music manifold setting, this could align:

- Different performances of the same piece.
- Different instrumentations of the same score.
- EEG responses of different subjects to the same musical stimulus.

### 4.4. Combined effect of the objectives

The joint loss:
\[
\mathcal{L}_{total} = \mathcal{L}_{recon} + \alpha \mathcal{L}_{geo} + \beta \mathcal{L}_{align}
\]

balances:

1. **Fidelity** (\(\mathcal{L}_{recon}\)): The latent manifold must encode enough information to reconstruct the input (EEG or music).
2. **Geometric faithfulness** (\(\mathcal{L}_{geo}\)): The manifold distances should reflect the structure of the input space.
3. **Cross-domain invariance** (\(\mathcal{L}_{align}\)): Latent representations of “the same” underlying state across different domains (subjects, modalities, performances) should be aligned.

This yields a manifold that is:

- Low-dimensional and structured (via Riemannian geometry),
- Dynamically coherent over time (via Neural ODE + sequence models),
- Geometrically meaningful (via \(\mathcal{L}_{geo}\)),
- And domain-agnostic (via \(\mathcal{L}_{align}\)).

---

## 5. Conceptual tie-in to a “music manifold”

Given the file name and project context, the same mathematics can be applied to a music manifold:

1. **Latent manifold \(\mathcal{M}\)**: Represents musical states (e.g., harmonic context, rhythm, timbre) with appropriate curvature:
   - Hyperbolic for hierarchical tonal relations (keys, chords).
   - Spherical for angular relationships (pitch-class circles, rhythmic cycles).

2. **Geodesic-aware attention**: Lets a Transformer attend along musically meaningful paths on \(\mathcal{M}\), e.g., focusing on harmonically close chords or rhythmically aligned positions.

3. **Neural ODE on \(\mathcal{M}\)**: Models continuous musical evolution, such as smooth modulation of key, gradual change in timbre, or tempo curves, as flows on the manifold.

4. **Geometric loss**: Ensures that musically similar inputs (e.g., similar chord progressions or motifs) are close on the manifold in geodesic distance.

5. **Contrastive alignment**: Aligns different musical realizations (performances, orchestrations, or subjects’ EEG responses to music) into a shared manifold representation.

---

If you’d like, I can next derive specific gradient expressions (e.g., for \(\mathcal{L}_{geo}\) on the Poincaré ball) or show how to implement the manifold ODE step explicitly for a chosen geometry.