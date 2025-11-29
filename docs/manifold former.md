Here are the core mathematical formulations and scientific embeddings defined in the **ManifoldFormer** framework.

### 1. Riemannian Latent Embedding
[cite_start]The model maps Euclidean EEG inputs to a latent manifold $\mathcal{M}$ using a reparameterization trick that enforces manifold constraints (Hypersphere or Hyperbolic)[cite: 104, 105, 106].

$$z=\Pi_{\mathcal{M}}(\mu+\epsilon\odot \exp(\sigma/2))$$
[cite_start][cite: 107]

* **Variables:**
    * [cite_start]$\mu, \sigma$: Mean and log-variance vectors from the encoder[cite: 105].
    * [cite_start]$\epsilon\sim\mathcal{N}(0,I)$: Gaussian noise[cite: 110].
    * [cite_start]$\Pi_{\mathcal{M}}(\cdot)$: Projection operator enforcing constraints (e.g., $||z||<1$ for Poincaré ball or $z \in \mathbb{S}^{d-1}$)[cite: 110, 111].

### 2. Geodesic-Aware Attention Mechanism
[cite_start]The Geometric Transformer computes attention weights based on geodesic structure rather than Euclidean distance[cite: 114, 115].

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - \lambda D_{geo}\right)V$$
[cite_start][cite: 116, 117, 118, 119]

* **Variables:**
    * [cite_start]$D_{geo}$: Geodesic distances on the manifold $\mathcal{M}$[cite: 120].
    * [cite_start]$\lambda$: Geometric regularization parameter[cite: 120].

### 3. Manifold-Constrained Neural ODE
[cite_start]Neural state evolution is modeled using Neural ODEs, augmented by Transformer and LSTM history[cite: 124, 125].

**ODE Function:**
$$\frac{dz(t)}{dt}=f_{\theta}(z(t),t,c(t))$$
[cite_start][cite: 125]

**Solver:**
$$\hat{z}_{t+1}^{ode}=ODESolver_{Dopris}(z_{t},f_{\theta})$$
[cite_start][cite: 125]

**State Transition:**
$$z_{t+1}=\hat{z}_{t+1}^{ode}\oplus Tf_{auto}(z_{\le t})\oplus LSTM_{bi}(z_{\le t})$$
[cite_start][cite: 125, 126, 127]

* **Variables:**
    * [cite_start]$c(t)$: Contextual features (Fourier time embeddings, Mamba SSM, ResNet)[cite: 129].
    * [cite_start]$\oplus$: Concatenation operator[cite: 130].

### 4. Geometric Learning Objectives
[cite_start]The model optimizes a joint loss function comprising reconstruction, geometric consistency, and alignment[cite: 132, 133].

**Total Loss:**
$$\mathcal{L}_{total}=\mathcal{L}_{recon}+\alpha\mathcal{L}_{geo}+\beta\mathcal{L}_{align}$$
[cite_start][cite: 133]

**Geometric Consistency Loss:**
[cite_start]Preserves local neighborhood structure by matching manifold distances to input Euclidean distances[cite: 134, 138].
$$\mathcal{L}_{geo}=\sum_{i,j}\left| \|z_{i}-z_{j}\|_{\mathcal{M}}-\|x_{i}-x_{j}\|_{2} \right|^{2}$$
[cite_start][cite: 134]

**Contrastive Alignment Loss:**
[cite_start]Enables cross-subject transfer[cite: 134, 138].
$$\mathcal{L}_{align}=-\sum_{i}\log\frac{\exp(sim(z_{i}^{(1)},z_{i}^{(2)})/\tau)}{\sum_{j}\exp(sim(z_{i}^{(1)},z_{j}^{(2)})/\tau)}$$
[cite_start][cite: 134]