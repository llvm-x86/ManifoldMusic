# NEXT: The Path to Resonance

**Status**: 🔴 **CRITICAL DEBUG MODE**  
**Objective**: Solve the Synthetic Oscillator (Oracle Distance < 0.05).  
**Directive**: Stop all feature development. Verify the core.

---

## 1. The Hard Constraint

We are currently blocked. The model cannot learn a simple synthetic oscillator (Oracle Distance stuck at ~0.25). Until this is resolved, all higher-level theories (RBIE, Music Manifold, Photonic Logic) are theoretical fiction.

**Gatekeeper**:
> **No new modules, no "music", no fancy manifolds until Oracle Distance < 0.05 on the synthetic oscillator.**

---

## 2. The Diagnosis: "Squashed" & "Phase Blind"

Our analysis of `internal/STATUS.md` and `GPT_NEXT.md` reveals two fundamental failure modes:

1.  **Squashing (Amplitude Death)**: The model minimizes MSE by outputting near-zero amplitude, killing the signal to play it safe.
    *   *Theoretical Fix*: **Log-Magnitude Spectral Loss** (enforces energy presence).
2.  **Phase Blindness (Geometry Mismatch)**: The model matches the frequency spectrum (magnitude) but fails to align the *phase* (rotation) with the Oracle.
    *   *Theoretical Fix*: **Phase Consistency Loss** + **Geometric Alignment** (Dynamic Rotation).

**However**, despite these theoretical fixes, the model is still failing. This suggests the implementation of these fixes is either buggy, conflicting, or fundamentally mis-specified.

---

## 3. The Plan: Rigorous Ablation & Reconstruction

We will execute the **"Patch A"** plan from `GPT_NEXT.md` to isolate the failure. We strip the model down to its atomic components and re-verify each layer.

### Phase I: The Trivial Baseline (Sanity Check)
*Goal: Prove the Decoder + Loss can learn ANYTHING.*

1.  **Patch A: Lobotomize the Model**
    *   **Encoder**: Deterministic, Euclidean (No VAE, No Sphere).
    *   **Dynamics**: Identity (No RK4, No Memory, No Kuramoto).
    *   **Loss**: Pure MSE.
    *   **Task**: Map `x[t]` to `x[t]` (Identity) or `x[t+1]` (Next Step).
    *   *Success Criteria*: MSE < 0.001 within 2 epochs.

2.  **Patch B: Spectral Sanity**
    *   **Setup**: Same as Patch A.
    *   **Loss**: Switch to **Spectral Loss Only** (Fractional + Phase).
    *   *Success Criteria*: Spectral Loss drops rapidly. Visual inspection shows signal preservation (no squashing).

### Phase II: Reintroducing Geometry
*Goal: Prove the Spherical VAE works.*

3.  [x] **Patch C: The Spherical Constraint**
    *   **Encoder**: Project output to Unit Sphere.
    *   **Dynamics**: Identity.
    *   **Loss**: Spectral + MSE.
    *   *Success Criteria*: Model can still learn to reconstruct/predict despite the bottleneck. (Result: Learning, but high cost. MSE 0.34, Spec 1.52)

### Phase III: Reintroducing Dynamics
*Goal: Prove the ODE Solver works.*

4.  [x] **Patch D: Minimal Dynamics**
    *   **Dynamics**: Simple Residual RNN Cell (No RK4).
    *   *Success Criteria*: Improvement over Identity baseline. (Result: Failed. Loss 1.93 > 1.86. Model prefers Identity.)

5.  [x] **Patch E: The Full Manifold Dynamics**
    *   **Dynamics**: Re-enable RK4, Power-Law Memory, Kuramoto.
    *   *Success Criteria*: Oracle Distance < 0.05. (Result: Success with Zero Init. Loss 1.79 < 1.86. Random Init Failed.)

---

## 4. Theoretical Alignment (The "Why")

While we debug, we keep the theoretical end-state in mind. This is what we are building *towards*, but we do not implement it until the foundation is solid.

### The "Three Pillars" of the Core
(From `internal/STATUS.md`)

1.  **Robust Spectral Loss**: The "Echolocation" signal. It must be un-gameable.
    *   *Key*: Log-Magnitude for energy, Phase-Consistency for geometry.
2.  **Spectral-Only Optimization**: The model must "grok" the geometry purely from resonance, not from pixel-matching (MSE).
    *   *Key*: Trust the physics. If the spectrum is right, the signal is right.
3.  **Consistent Geometry**: The Decoder must use the *aligned* latent state (`z_dyn`), not the raw encoder output.
    *   *Key*: The "Rotation" is part of the manifold flow.

### The RBIE Connection
(From `docs/riemann_information_extraction_derivation.md`)

*   **Pseudo-Zeros**: The "quiet" parts of the spectrum define the structure.
*   **Imaginary Entropy**: High entropy = Noise, Low entropy = Structure.
*   *Future Application*: Once the oscillator is working, we will use RBIE to *mask* the input to the ManifoldFormer, feeding it only the "structural" frequencies. This will be the "Attention" mechanism of the future.

### The Photonic Logic
(From `docs/zyra_photon_derivation.md`)

*   **Field Dynamics**: The latent space is a "field" where patterns self-organize.
*   *Future Application*: The "Consensus" mechanism (SupremeQueen) will be used to stabilize the *long-term* predictions of the ManifoldFormer, voting on the "key" or "genre" of the generated music.

---

## 5. Immediate Action Items

1.  [x] **Create `debug_ablation.py`**: A script to run Phase I (Patch A/B) without modifying the main codebase destructively.
2.  [x] **Run Patch A**: Verify Euclidean MSE learning. (Result: Converged to 0.0071 in 5 epochs)
3.  [x] **Run Patch B**: Verify Spectral Loss learning. (Result: Converged to 0.9560 in 5 epochs)
4.  [x] **Report**: Update `internal/STATUS.md` with results. (Done. Root cause identified: Random Init. Fix applied: Zero Init.)

**DO NOT PROCEED TO PHASE II UNTIL PHASE I IS GREEN.**
