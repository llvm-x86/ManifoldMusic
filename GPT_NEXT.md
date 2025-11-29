# Recommended Next Path (from GPT-5.1)

```markdown
# GPT_NEXT: Debug Plan for Synthetic Oscillator Failure

Status from `internal/STATUS.md`: **FAILED / STUCK** on the *simplest* synthetic task.  
Oracle Distance flat ≈ 0.25, Spectral Loss nowhere near 0.04.  
We do **not** have a working core. No more features. We debug.

Below is a concrete, minimal, *ordered* plan.

---

## 0. Hard Constraint

Until Oracle Distance < 0.05 on the synthetic oscillator:

- No new modules.
- No RBIE, no “music”, no fancy manifolds.
- Only: verify that **this** code can learn **this** toy problem.

---

## 1. Sanity Check: Can the Decoder + Loss Learn a Trivial Mapping?

### 1.1 Kill the VAE Noise and Dynamics

Right now the gradients are going through:
- stochastic VAE sampling
- manifold exponential
- Kuramoto + power-law memory
- RK4 integration

That’s a lot of moving parts. We don’t even know if the decoder + spectral loss alone can fit anything.

**Patch A (in `ManifoldFormer.__call__`)**:

Temporarily bypass dynamics and sampling:

```python
def __call__(self, x, rng):
    # --- Encoder, but deterministic and Euclidean ---
    h = nn.relu(self.vae.encoder_conv1(x))
    h = nn.relu(self.vae.encoder_conv2(h))
    mu_raw = self.vae.encoder_dense(h)
    # NO normalization, NO sigma, NO sampling
    z = mu_raw  # purely Euclidean latent

    # NO dynamics
    z_dyn = z

    # Decode
    out = self.vae.decode(z_dyn)

    # For compatibility with loss_fn, fake values
    recon_vae = out
    mu = z
    sigma = jnp.ones_like(z)

    return out, recon_vae, z, z_dyn, self.iso_scale, mu, sigma
```

Also in `loss_fn`:

- Set `l_kl = 0.0`.
- Optionally set `l_iso = 0.0`, `l_smooth = 0.0` (we’re not using them anyway).

### 1.2 Make the Loss a Simple MSE First

Before spectral nonsense, verify plain supervised learning works.

In `loss_fn`:

```python
# Replace spectral_loss definition
spectral_loss = jnp.mean((pred - y_pred_target) ** 2)
```

Train for 2–3 epochs with this setup.

**Expected**:
- Training MSE should drop sharply (well below 0.1) on this tiny synthetic dataset.
- If it doesn’t, something is fundamentally broken (data shapes, optimizer, learning rate, batching).

If MSE does not fall:

- Print shapes at each stage (encoder input, z, out).
- Check that `x` and `y_pred_target` are what we think (no off-by-one mistake).
- Verify optimizer actually updates params (inspect a parameter norm across steps).

**Do not proceed** until this trivial regime learns.

---

## 2. Step-by-Step: Reintroduce Complexity with Unit Tests

Once plain MSE + simple encoder/decoder works, add components back *one at a time* with explicit tests.

### 2.1 Re-enable Spectral Loss, Still No Manifold/Dynamics

Keep `z = mu_raw` (Euclidean), no dynamics.

In `loss_fn`:

```python
spectral_loss = 1.0 * l_frac  # only fractional_fidelity_loss
```

Leave entropy term off for now.

Run 2–3 epochs:

- Check `l_frac` actually decreases.
- Plot `pred[0, :, 0]` vs `y_pred_target[0, :, 0]` for a fixed batch/sample index across epochs.

If spectral loss can’t be minimized in this simple setting, the issue is in `fractional_fidelity_loss` (or targets), not geometry.

### 2.2 Deterministic Latent: VAE Without Noise, Still Euclidean

Turn VAE back on but deterministic:

In `VAE.__call__`:

- Compute `mu_raw` as now.
- **Do not** sample epsilon; set `z = mu = mu_raw` (no normalization, no exp map, no sigma).

Confirm training still converges.

If enabling this breaks learning, the encoder itself is mis-specified (conv/shape issues).

---

## 3. Isolate the Manifold: Sphere Without Dynamics

We need to know if the spherical VAE + decoder is trainable *at all*.

### 3.1 Turn On Spherical Constraint, Still No Dynamics

In `VAE.__call__`:

- Keep `mu` normalized to the sphere.
- Set `z = mu` (no tangent noise, no exp map).

So:

```python
mu_raw = self.encoder_dense(h)
mu = mu_raw / jnp.maximum(jnp.linalg.norm(mu_raw, axis=-1, keepdims=True), 1e-8)
z = mu
```

No sigma, no sampling.

Train with **spectral loss only** (or even MSE first). If this doesn’t learn, the decoder cannot invert the spherical embedding. That’s a clear bottleneck.

If it works, then the sphere is not the problem; the dynamics are.

---

## 4. Minimal Dynamics Debug: Single-Step, No Memory, No Kuramoto

Right now `Dynamics.__call__` is a complicated RK4 over:

- group-conv “memory_force” that doesn’t even use the fractional kernel argument
- Sakaguchi-Kuramoto attention over full sequence
- tangent projection + retraction

We need to know if *any* dynamics step can be learned.

### 4.1 Replace Dynamics with a Simple Residual RNN Cell

Temporarily:

```python
class Dynamics(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, z):
        # z: (B, T, D)
        W = self.param('W', nn.initializers.xavier_uniform(), (self.latent_dim, self.latent_dim))
        b = self.param('b', nn.initializers.zeros, (self.latent_dim,))

        # One-step residual update
        z_next = z + jnp.tanh(jnp.einsum('btd,df->btf', z, W) + b)
        return z_next
```

Keep the encoder as spherical (or Euclidean) and decoder as is.

Train with spectral loss (or MSE) and check if:

- `pred` improves over `y_pred_target`.
- Oracle Distance (even if not perfect) moves in the right direction.

If even this simple residual dynamics doesn’t help vs. identity, the issue is not geometry; it’s the training setup (targets, time indexing).

---

## 5. Check the Time Indexing and Targets

Current dataset:

```python
# Input: Noisy [0:-1]
# Target: Clean [1:]
# Oracle: [0:-1]
```

Model:

- Takes `x` (shape `(B, T, C)` where `T = seq_len - 1`).
- Produces `pred` with same `T`.
- Loss compares `pred` to `y_pred_target` (clean [1:]).

**Potential bugs**:

1. The model might be learning to map `x[t]` → `y[t]` (next step) but the dynamics are effectively mapping `z[t]` → `z_next[t]` misaligned with that.

2. `Dynamics` is applied to the entire sequence in parallel; there is no explicit notion of “t → t+1” in the loss.

**Sanity test**:

- Change dataset to trivial **identity** task:

  ```python
  return noisy_signals[:, :-1, :], clean_signals[:, :-1, :], ..., oracle_trajectories[:, :-1, :]
  ```

  and train the simplified model (no dynamics, Euclidean z) with MSE.

- If that works, then reintroduce the one-step prediction target and confirm that a trivial model (no dynamics) can also learn `x[t] → y[t+1]`. If not, we mis-specified the target or architecture for next-step prediction.

---

## 6. Gradient and Scale Diagnostics

Once you have a configuration that *should* learn but doesn’t, inspect gradients and magnitudes.

### 6.1 Check Gradient Norms

Add a debug function:

```python
def debug_grad_norms(grads):
    flat, _ = jax.tree_util.tree_flatten(grads)
    norms = [jnp.linalg.norm(g) for g in flat if g is not None]
    print("Grad norms: min", float(jnp.min(jnp.array(norms))),
          "max", float(jnp.max(jnp.array(norms))))
```

Call it inside `train_step` after `jax.grad`:

```python
grads = jax.grad(loss_fn)(...)
debug_grad_norms(grads)
```

If norms are ~0 or NaN, we have vanishing/exploding gradients (likely from log/exp or arccos).

### 6.2 Clip or Remove Risky Terms During Debug

In `loss_fn`:

- Temporarily **remove** entropy term, KL, isometry, smoothness. Only keep the simplest loss you are testing (MSE or l_frac).
- Ensure no `arccos` or `log` is in the active loss during basic debugging.

---

## 7. Rebuild the Intended Model Incrementally

Only after the simplified pieces are confirmed to learn:

1. Re-enable sphere constraint in encoder.
2. Re-enable VAE sampling (but keep sigma small or annealed).
3. Replace residual dynamics with a *very* minimal manifold-aware step:

   ```python
   def __call__(self, z):
       # z: (B, T, D) on sphere
       W = self.param('W', nn.initializers.xavier_uniform(), (self.latent_dim, self.latent_dim))
       v = jnp.einsum('btd,df->btf', z, W)
       # project to tangent
       dot = jnp.sum(v * z, axis=-1, keepdims=True)
       v_tan = v - dot * z
       # small step retraction
       h = 0.1
       z_new = z + h * v_tan
       z_new = z_new / jnp.maximum(jnp.linalg.norm(z_new, axis=-1, keepdims=True), 1e-8)
       return z_new
   ```

4. Only when this learns, start layering in:
   - power-law memory (but *actually use* `get_fractional_kernel`),
   - Kuramoto attention,
   - RK4.

At each step, re-run the synthetic oscillator and watch Oracle Distance. If it jumps back to ~0.25 and stays flat, the last component you added is the culprit.

---

## 8. Concrete Immediate TODO List

1. Implement **Patch A** (no dynamics, no sampling, Euclidean z).
2. Switch loss to pure MSE and verify rapid convergence on the synthetic dataset.
3. If that works, re-enable spectral loss (no manifold, no dynamics) and verify it can be minimized.
4. Then reintroduce sphere constraint without dynamics and confirm training still works.
5. Replace dynamics with a trivial residual cell and test again.
6. Only after these pass, start reintroducing the current manifold + RK4 dynamics piece by piece.

No more “promised land” talk until step 2 is green. Right now we don’t even have a working linear path from noisy input to clean next-step output under a simple loss; everything else is noise.

```