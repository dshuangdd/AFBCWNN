# AFBCWNN: Adaptive Frequency-Based Constructive Wavelet Neural Network

Official implementation of "Adaptive Frequency-Based Constructive Wavelet Neural Network for Nonlinear Dynamic Systems"

## Overview

This repository contains the official implementation of **AFBCWNN**, a novel adaptive neural network architecture for tracking control of unknown nonlinear dynamic systems. The key innovation is leveraging frequency-domain analysis to guide network structure construction and parameter adaptation, achieving superior tracking accuracy with significantly fewer parameters compared to existing methods.

### Key Features 

-  **Frequency-guided structure initialization**: Automatically determines optimal initial network structure based on energy distribution
-  **Dynamic structure adaptation**: Incrementally adds high-energy wavelet bases and prunes redundant nodes
-  **Lyapunov-based stability**: Guarantees bounded tracking error and closed-loop stability
-  **Computational efficiency**: Achieves comparable or better performance than many advanced methods
-  **Robustness**: Maintains performance under external disturbances

##  Repository Structure
```
AFBCWNN/
├── Section_IV-A/                    # Second-order system experiments
│   ├── AFBCWNN_2D_INIT_M.py        # Initial resolution determination of Algorithm 1
│   └── AFBCWNN_2D_MAIN.py          # Main control loop of Algorithm 1 (tunable λ, β)
│
├── Section_IV-B/                    # Fourth-order system comparisons
│   ├── AFBCWNN_4D_INIT_M.py        # Initial resolution determination of Algorithm 1
│   ├── AFBCWNN_4D_MAIN.py          # Main control loop of Algorithm 1 
│   ├── ACWNN.py                     # Adaptive Constructive WNN baseline
│   ├── FBCWNN.py                    # Frequency-based CWNN (non-adaptive)
│   ├── RBF.py                       # Adaptive RBF-NN (256/625 nodes)
│   ├── SAC.py                       # Soft Actor-Critic (model-free)
│   └── TRANSFORM.py                 # Transformer-based policy (model-free)
│
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

##  Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- NumPy, SciPy, Matplotlib

### Setup
```bash
# Clone the repository
git clone https://github.com/dshuangdd/AFBCWNN.git
cd AFBCWNN

# Install dependencies
pip install -r requirements.txt
```

##  Quick Start

### Section IV-A: Parameter Sensitivity Analysis (2D System)

Reproduce Figure 2 by varying control gains `(λ, β)`:
```python
# Edit parameters in AFBCWNN_2D_MAIN.py
lambda_val = 10.0  # Options: 10.0, 20.0
beta_val = 10.0    # Options: 10.0, 20.0

python Section_IV-A/AFBCWNN_2D_MAIN.py
```

**Available configurations**: `(10, 10)`, `(10, 20)`, `(20, 10)`, `(20, 20)`

**Note**: These experiments demonstrate the influence of parameters λ and β on AFBCWNN performance. All configurations achieve convergence of the augmented tracking error to the predefined accuracy, but with different transient responses and parameter efficiency (see Tables I-II and Figure 3 in the paper).

### Section IV-B: Comparative Experiments (4D System)

#### Step 1: Determine Initial Resolution
```bash
python Section_IV-B/AFBCWNN_4D_INIT_M.py
```

This step computes the optimal starting resolution `m*` based on energy estimation (Algorithm 1, lines 6-21).

**Mechanism**: By progressively increasing resolution and computing subspace energy, when E̅ₘ₋₁ ≥ Êₘ, the main energy is concentrated at resolution m-1, thus determining the initial structure.

#### Step 2: Run Main Control Loop
```python
# AFBCWNN with/without disturbance
python Section_IV-B/AFBCWNN_4D_MAIN.py

# Enable disturbance at t=120s (default: disabled)
# Edit in AFBCWNN_4D_MAIN.py:
enable_disturbance = True  # Adds d(t) = 6sin(4t)
```

**Disturbance mechanism**: The disturbance d(t) = 6sin(4t) is injected into the fourth state equation at t=120s to test algorithm robustness. AFBCWNN, ACWNN, and RBF codes all support this disturbance toggle.

#### Compare with Baselines
```bash
# Adaptive methods comparison
python Section_IV-B/ACWNN.py          # ACWNN baseline
python Section_IV-B/FBCWNN.py         # Non-adaptive FBCWNN
python Section_IV-B/RBF.py            # RBF-NN (toggle RBF_NODES = 256 or 625 in code)

# Model-free methods comparison
python Section_IV-B/SAC.py            # Soft Actor-Critic reinforcement learning
python Section_IV-B/TRANSFORM.py      # Transformer sequence policy
```

**RBF structure options**:
- `RBF_NODES = 256`: Uses 4×4×4×4 grid centers (moderate complexity)
- `RBF_NODES = 625`: Uses 5×5×5×5 grid centers (high complexity)

These two structures demonstrate that fixed-structure adaptive neural networks cannot guarantee desired tracking performance even with more parameters.

##  Configuration Guide

### Core Hyperparameters (Algorithm 1)

| Parameter | Description | Typical Range | Default(2D/4D) |
|-----------|-------------|---------------|---------|
| `μ` | Energy separation factor | (0, 1) | 1/3 |
| `ς` | Sampling ratio for initial basis | (1/3, 2/3) | 0.36 |
| `δ_ac` | Target tracking accuracy | - | 0.12/0.05 |
| `δ°_ac` | Auxiliary accuracy (stricter) | - | 0.07/0.04 |
| `ΔT` | Dwell time (seconds) | - | 120 |
| `N_d` | Termination counter | [1, 3] | 1 |
| `ξ_r` | Pruning threshold | - | 0.1/0.02 |
| `ξ_h` | Freezing threshold | - | 0.1/0.02 |
| `λ` | Augmented error gain | [2, 20] | 20.0/2.0 |
| `β` | Feedback gain | [10, 20] | 20.0/10.0 |

**Parameter Guidelines**:
- **μ**: Controls the proportion of bases added per expansion. Smaller values allow finer-grained expansion (fewer bases per step), while larger values accelerate convergence but may introduce redundancy.
- **ς**: Determines the initial basis density in the coarsest subspace W₁. The range (1/3, 2/3) balances coverage and computational efficiency.
- **δ°_ac**: Auxiliary accuracy threshold, stricter than the final target δ_ac, used to ensure robust performance before triggering structural decisions.
- **ΔT**: Dwell time window during which the system runs before adding new bases. Doubles (ΔT ← 2ΔT) after successfully meeting δ°_ac.
- **N_d**: Number of consecutive successful dwell periods required before algorithm terminates expansion. Values between 1-3 balance convergence verification and computational efficiency.
- **ξ_r, ξ_h**: Pruning and freezing thresholds. ξ_r removes bases with negligible weights, ξ_h freezes bases with minimal weight variation.
- **λ, β**: Control law gains. β affects transient response speed, λ affects steady-state accuracy (see Section IV-A analysis).

### Disturbance Settings
```python
# In AFBCWNN_4D_MAIN.py / ACWNN.py / RBF.py
enable_disturbance = True  # Toggle disturbance injection
disturbance_start_time = 120.0  # Start time (seconds)

def disturbance(t):
    return 6 * np.sin(4 * t) if t >= disturbance_start_time else 0.0
```

The disturbance takes the form of a sinusoidal wave d(t) = 6sin(4t), injected into the fourth state equation at t=120s:
```
ẋ₄ = f(x) + u(t) + d(t)·1[t≥120](t)
```

### RBF Network Configurations
```python
# In RBF.py, choose one:
RBF_NODES = 256   # 4×4×4×4 grid (moderate complexity)
RBF_NODES = 625   # 5×5×5×5 grid (high complexity)
```

## Algorithm Overview

### Algorithm 1: Incremental Module in AFBCWNN

**Phase 1: Initial Resolution Determination** (lines 6-21)
```
1. Start with coarsest subspace W₁
2. Run system for dwell time ΔT
3. Compute energy estimate Êₘ via adaptive law
4. Calculate EMA: E̅ₘ = (αE̅ₘ₋₁ + (1-α)Êₘ) / (1-αᵐ)
5. If E̅ₘ₋₁ ≥ Êₘ, set m* = m-1 and exit
6. Else, expand to Wₘ₊₁ and repeat
```

**Explanation**: This phase leverages Assumption 2 (unimodal energy) to find the main energy subspace. Once energy starts decreasing, the previous resolution is the optimal initial structure.

**Phase 2: Main Control Loop** (lines 22-54)
```
1. Initialize with bases from Vₘ* and Wₘ*
2. While N_counter < N_d:
   a. Run system for ΔT and update weights
   b. Compute MATEDT: δ̄ᵢ = max|δ(t)| over [iΔT, (i+1)ΔT]
   c. If δ̄ᵢ ≤ δ°_ac:
      - Increment N_counter
      - Double dwell time: ΔT ← 2ΔT
   d. Else:
      - Reset N_counter
      - Add top-μ̄ energy bases from Wₘ
      - Expand to nearest 2^d bases in Wₘ₊₁
      - Increase μ̄ if all bases added
3. Prune bases with |θ̂ₘ,ₙ| < ξ_r
4. Freeze bases with Γ(θ̂ₘ,ₙ) < ξ_h
```

##  Related Work

- **FBCWNN** (offline, static): [Optimizing Basis Function Selection in Constructive Wavelet Neural Networks](https://arxiv.org/abs/2507.09213)
- **ACWNN** (online, no frequency guidance): J.-X. Xu and Y. Tan, "Nonlinear adaptive wavelet control using constructive wavelet networks," IEEE Trans. Neural Networks, 2007


##  Foundation

- Wavelet theory foundations: Mallat's multiresolution analysis
- Adaptive control framework: Lyapunov-based stability theory
- Baseline implementations: PyWavelets, PyTorch

---

** Note**: This is a research prototype. For production use, additional testing and validation are recommended.
