# Operation Promised Land: Geometric Supervision

## Problem Diagnosis
We are "playing whack-a-mole" because we are trying to induce a specific latent geometry (the "Oracle" hypersphere trajectory) using only indirect, unsupervised signals (reconstruction loss, smoothness, ad-hoc entropy). This leads to:
1.  **Instability**: Conflicting losses (e.g., Decorrelation vs. Coupling).
2.  **Cheating**: Flexible modules (Dynamic Rotation) finding trivial solutions that minimize spectral loss without true geometric alignment.
3.  **Drift**: The model "stands" (stable) but doesn't "run" (converge to 0.02) because there is no strong gradient pointing to the exact Oracle manifold.

## The Solution: Direct Geometric Supervision
Since we have the **Oracle Trajectory** (the ground truth analytic signal on the sphere) available during training, we should **explicitly supervise** the latent space geometry.

Instead of hoping the VAE finds the "Promised Land" in the dark, we will give it a map.

## The Plan

### 1. Differentiable Procrustes Loss
We will implement a differentiable version of the Procrustes alignment inside the JAX `loss_fn`.
*   **Mechanism**: $L_{geo} = \min_R \| Z_{learned} R - Z_{oracle} \|_F^2$
*   **Effect**: This forces the Encoder to output a trajectory that is **isometric** to the Oracle trajectory. It allows for global rotation (which is physically valid) but penalizes any distortion of the shape (squashing, jitter, wrong frequencies).

### 2. Simplify the Stack
With strong geometric supervision, we can remove the "band-aids":
*   **Remove/Simplify Dynamic Rotation**: The Encoder should learn to target the canonical frame directly (or a stable frame aligned via Procrustes). We can keep a simple static or rigid alignment if needed, but the heavy lifting will be done by the supervised loss.
*   **Trust the Physics**: Keep the Power-Law Memory and Kuramoto Dynamics. They are correct. The supervision will ensure they receive the clean inputs they need.

### 3. "Zyra" Spectral Entropy
We will retain the **Power-Based Spectral Entropy** (from `reference/zyra_photon.py`) as a regularizer to ensure the signal remains "focused" (low entropy, clear harmonics) and doesn't degenerate into noise, even if the Procrustes loss is fighting hard.

## Implementation Steps
1.  **Port `procrustes_distance` to JAX**: Make it differentiable and batched.
2.  **Update `loss_fn`**: Add `l_oracle` with a high weight (e.g., 1.0) to drive alignment.
3.  **Train**: Expect `Oracle Distance` to drop rapidly as the gradient now directly optimizes it.
