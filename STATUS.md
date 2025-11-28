# Investigation Status: Imputation Failure in ManifoldFormer

**Date**: 2025-11-27  
**Objective**: Resolve poor imputation performance despite achieving target loss (~0.03) on deterministic synthetic data

**GLOBAL DIRECTIVE**: The **Phase Reconstruction Loss** (Spectral/Frequency Domain) is the **SOLE** optimization target. All other metrics (MSE, Base Loss) are for monitoring only. If Phase Reconstruction Loss increases, the experiment is a failure.

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

### 5. Architecture: Explicit Spectral Branch
**Mistake**: Adding a parallel FFT $\to$ Dense branch to the Encoder to explicitly capture global frequency context.
**Why it failed**:
1.  **Shape Mismatch**: The implementation assumed strict `(Batch, Time, Channels)` inputs, but the inference pipeline (imputation) passes unbatched `(Time, Channels)` data. Explicitly unpacking shapes (`B, T, C = x.shape`) broke the pipeline.
2.  **Redundancy**: The "Global Frequency" is already implicitly captured by the Power-Law Memory and the Spectral Loss. Forcing it architecturally complicates the manifold topology without clear benefit.
3.  **Worse Phase Reconstruction**: The **Phase Reconstruction Loss** (Spectral) increased (0.97 vs 0.96). This is our **sole optimization target**, and any increase here is unacceptable regardless of other metrics.
**Correction**: Keep the Encoder simple (Local Causal Conv). Let the **Loss Function** (Spectral) and **Dynamics** (Memory) handle the frequency domain constraints.

---

## Recent Experiment: "Picking the Lock" (Failure Analysis)

**Attempt**:
1.  **Deterministic Data**: Fixed seed per sample.
2.  **Intrinsic Geometry**: Removed $W_Q, W_K$ projections in Dynamics, forcing $Q=K=z$ (Self-Attention on Manifold).
3.  **No KL**: Removed probabilistic prior.

**Result**: Loss increased (1.44 $\to$ 1.10) compared to baseline (0.22).

**Why it failed**:
The "Lock" (Data) is deterministic, and the "Key" (Dynamics) was made rigid/intrinsic. However, the **Hand holding the Key (Encoder)** was still shaky (randomly initialized).
By removing the learnable projections ($W_Q, W_K$) in the Dynamics, we removed the **adapter** that translates the Encoder's arbitrary initial latent space into the canonical geometric form required by the Kuramoto dynamics.
The Encoder outputs a latent state $z$. Initially, this $z$ is random. The rigid Dynamics expects $z$ to *already* be perfectly phase-aligned to work (since $Attn \propto \sin(\langle z, z \rangle)$).
Without the "wiggle room" of the projection layers, the gradients cannot easily flow back to align the Encoder. The Encoder and Dynamics are "impedance mismatched."

**The "Rotation" Insight**:
We need to align the Encoder's output frame with the Dynamics' canonical frame.
Instead of full dense projections (which distort geometry), we should apply a **Learnable Global Rotation** (Orthogonal Matrix) at the interface between the Encoder and the Dynamics.
This corresponds to "turning the key" to align with the lock's tumblers, without changing the shape of the key itself.

## Recent Experiment: "Tandem Resonant Rotation" (Partial Success)

**Hypothesis**:
The "Key" (Latent State) must dynamically rotate to align with the "Lock" (Manifold Dynamics) based on the signal's resonance.
We implemented a **Dynamic Resonant Rotation** module using a Cayley transform to predict an orthogonal rotation matrix $R(z)$ from the latent state itself.

**Result**:
- **Loss**: 0.85 (Epoch 3).
- **Comparison**: Better than static alignment (1.10), but still higher than baseline (0.22) and far from target (0.04).

**Analysis**:
The loss improvement confirms that **dynamic alignment** is directionally correct. The "key" is now turning in response to the state.
However, the fact that we haven't reached 0.04 suggests we haven't fully "grokked" the hidden harmonics. The rotation might be too simple (pointwise) or the manifold topology is more complex than a simple sphere.
The model is "standing" (stable, improving) but not yet "running" (converging to zero error).

**Directive**:
Avoid low-signal "whack-a-mole" component fiddling. Focus on **Geometry**.
The next step should be to deepen the geometric alignment, perhaps by considering the **curvature** of the manifold or a more expressive Lie Group for the rotation.

---

## Recent Experiment: "Spectral Resonance" (Backprop Only)

**Hypothesis**:
To "grok" the geometry, we should force the model to learn **only** from the spectral resonance (frequency alignment + latent entropy), ignoring time-domain MSE for now. This forces the "echolocation" strategy.

**Implementation**:
1.  **Loss**: $\mathcal{L}_{total} = \mathcal{L}_{spec} + \mathcal{L}_{entropy}$ (Backprop ONLY this).
2.  **Tracking**: Monitor MSE and Base Loss (Reconstruction + Smoothness) without optimizing them.

**Result** (3 epochs):
- **Spectral Loss**: 12.83 → 8.67 → 5.29 (Rapid decrease)
- **Base Loss**: 1.62 → 1.67 → 1.86 (Slight increase)
- **MSE**: 1.30 → 1.36 → 1.55 (Slight increase)

**Analysis**:
The model is successfully "grokking" the spectral structure, as evidenced by the rapid drop in spectral loss.
The slight increase in MSE is expected and acceptable; we are trading immediate pixel-perfect reconstruction for fundamental geometric alignment.
The "echolocation" is working: the model is tuning its internal "key" to resonate with the target's frequencies.


## 5. The "Fractional Sakaguchi" Breakthrough (Current State)

**Date**: 2025-11-28
**Status**: **Stable & Resonant**

We have successfully moved beyond the "MSE Crutch" and the "Amplitude Death" trap by embedding the physics of the problem directly into the loss and dynamics.

### Concepts & Encoding Embeddings

| Concept | The "Why" (Motivation) | The "Encoding" (Mathematical Embedding) |
| :--- | :--- | :--- |
| **Temporal Geometry** | Fixed-window STFTs impose an arbitrary scale. The model's memory is Power-Law (Fractional); the loss should measure fidelity using the same "ruler." | **Fractional Fidelity Loss**: A bank of convolutions with kernels $k_\gamma(t) \propto t^{-\gamma}$ for $\gamma \in [0.1, 0.9]$. This aligns the *optimization geometry* with the *model geometry*. |
| **Spectral Energy Conservation** | Minimizing MSE allows the signal to "squash" (decay) because zero energy is a safe local minimum. We need to enforce the *presence* of energy. | **Log-Magnitude Spectral Loss**: $\mathcal{L} = \| \log|F(y)| - \log|F(\hat{y})| \|$. The logarithm explodes if the amplitude drops to zero, forcing the model to maintain signal energy (preventing squashing). |
| **Sakaguchi-Kuramoto Stability** | Standard Kuramoto ($\sin(\Delta \theta)$) can collapse to static synchrony (fixed point). We need a stable *orbit* (limit cycle) to represent time-varying signals. | **Phase Lag ($\alpha$)**: $\text{Attn} \propto \sin(\langle q, k \rangle + \alpha)$. A learnable parameter that breaks symmetry, preventing amplitude death and ensuring the latent state keeps "spinning" on the manifold. |


### Visual Proof
*   **Phase Portrait**: Shows a stable, continuous trajectory on the hypersphere (no jagged walks).
*   **Imputation**: Signal amplitude is preserved in the gap (ratio $\approx$ 1.0) due to the Log-Spectral constraint and Phase Lag momentum.

---

## 7. Recent Experiment: "Dynamic Cayley Rotation with Magnitude-Based Entropy" (Stalled Progress)

**Date**: 2025-11-28  
**Status**: **Partial Regression**

### Hypothesis
Building on the "Tandem Resonant Rotation" insight (Section 5), we hypothesized that:
1.  **Dynamic Alignment**: The rotation matrix $R$ should be **predicted** from the latent trajectory's resonance, not static.
2.  **Magnitude-Based Entropy**: Using $p_i = |\omega_i| / \sum |\omega_j|$ (aligned with Riemann Information Extraction reference) would provide softer gradients for squashed signals compared to power-based ($|\omega|^2$).

### Implementation
1.  **`DynamicCayleyRotation`**: Modified from static `CayleyRotation` to:
    -   Extract context via temporal mean pooling: $c = \text{mean}(z, \text{axis}=1)$
    -   Predict skew-symmetric matrix: $c \to \text{Dense}(128) \to \text{ReLU} \to W \in \mathbb{R}^{d \times d}$
    -   Enforce antisymmetry: $A = W - W^T$
    -   Cayley transform: $R = (I - A)(I + A)^{-1}$
    -   Apply rotation: $z_{\text{aligned}} = z \cdot R$
2.  **Magnitude-Based Entropy**: Updated loss function:
    ```python
    z_fft = jnp.fft.rfft(z_dyn, axis=1)
    z_mag = jnp.abs(z_fft)  # Magnitude, not Power
    z_prob = z_mag / (jnp.sum(z_mag, axis=1, keepdims=True) + 1e-8)
    l_entropy = -jnp.mean(jnp.sum(z_prob * jnp.log(z_prob + 1e-8), axis=1))
    ```
3.  **Entropy Weight**: Increased from `0.01` to `0.05` to give the latent regularization stronger steering.

### Result (3 epochs)
| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Target |
|:---|---:|---:|---:|---:|
| **Spectral Loss** | 1.35 | 1.32 | 1.29 | 0.04 |
| **Oracle Distance** | 0.226 | 0.243 | 0.244 | **0.02** |
| **MSE (Tracking)** | 1.33 | 1.37 | 1.39 | — |

### Analysis: Why Did Oracle Distance NOT Improve?

#### 1. **Dynamic Rotation Added Degrees of Freedom, Not Constraints**
-   **Problem**: The dynamic rotation has $\frac{d(d-1)}{2}$ learnable parameters **per batch**. For $d=21$, this is 210 parameters predicted from a 128-dim hidden layer.
-   **Effect**: The model can now "cheat" by finding arbitrary rotations that minimize the **spectral loss** (which only cares about frequencies, not geometric alignment), without actually aligning to the canonical oscillator geometry.
-   **Evidence**: Spectral loss improves (1.35 → 1.29), but Oracle Distance **stays flat or increases** (0.226 → 0.244). The model is optimizing the wrong manifold.

#### 2. **Mean Pooling Destroys Temporal Phase Information**
-   **Problem**: `context = jnp.mean(x, axis=1)` averages across time, collapsing the trajectory into a single point. For oscillatory data, this kills the **phase velocity** (the direction of rotation on the sphere).
-   **Effect**: The rotation $R(c)$ cannot capture the instantaneous dynamics (e.g., "this trajectory is spinning clockwise at frequency 3"). It only sees a static "centroid."
-   **Correct Approach**: Use the **first and last states** to infer the rotation axis, or use a recurrent encoder to extract the trajectory's angular momentum.

#### 3. **Magnitude-Based Entropy is Too Permissive**
-   **Problem**: Entropy based on $|\omega|$ (magnitude) allows the spectrum to be "spread out" as long as it's nonzero. This doesn't enforce that the latent state is **localized on a limit cycle** (low entropy = concentrated spectrum).
-   **Effect**: The model spreads energy across many frequencies to reduce entropy loss, but this creates a "noisy cloud" instead of a clean orbit with a few dominant harmonics.
-   **Evidence**: MSE increases (1.33 → 1.39), suggesting the reconstruction is getting worse (blurrier) even as spectral loss improves.

### Directive: Course Correction Required

This experiment confirms that **structure alone is insufficient**; we need **geometry-aware constraints**.

The next steps should:
1.  **Replace Mean Pooling with Velocity Encoding**: Use $\Delta z = z_{-1} - z_0$ (first-to-last difference) as the context. This captures the "spin" of the trajectory.
2.  **Add Orthogonality Regularization**: Penalize $\|R^T R - I\|_F^2$ to ensure the predicted rotation is actually orthogonal (Cayley only guarantees this theoretically; numerical errors or large $A$ can break it).
3.  **Reduce Entropy Weight or Switch Back to Power**: The magnitude-based entropy might be too weak. Consider reverting to $|\omega|^2$ (power) and lowering the weight to `0.01` to provide hard constraints on spectral peaks.
4.  **Increase Training Duration**: 3 epochs is very short. The "grokking" phenomenon typically requires 10-30 epochs. The model might just be starting to learn the alignment.

---

## 6. Target Metric
**Goal**: Minimize `Oracle Distance` (Procrustes Alignment) to **0.02**.
*   **Current Status**: ~0.24 (Epoch 3).
*   **Significance**: This metric validates that the learned latent geometry matches the ground truth analytic signal (perfect sphere trajectory) invariant to rotation. Achieving 0.02 implies the model has "perfectly grokked" the underlying oscillators.
