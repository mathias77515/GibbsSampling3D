# GibbsSampling3D

A modular framework for **high-dimensional Gibbs sampling** applied to **linear Gaussian inverse problems**, with efficient use of **sparse linear algebra** and **Conjugate Gradient (CG) solvers**.

---

## Overview

This repository implements a Gibbs sampling approach for large-scale Bayesian inference problems of the form:

$$
\[
d = Hx + \epsilon, \quad \epsilon \sim \mathcal{N}(0, W^{-1})
\]
$$

with a Gaussian prior:

$$
\[
x \sim \mathcal{N}(\mu, C_x)
\]
$$

The posterior is explored using a Gibbs-style sampler where each iteration requires solving a large sparse linear system using CG methods.

This framework is designed for **scientific computing applications** such as:
- Cosmological map reconstruction
- 3D inverse problems
- Gaussian random field inference
- Large-scale linear Bayesian models

---

## Key Features

- 🚀 Scalable Gibbs sampling for high-dimensional problems  
- ⚙️ Sparse linear algebra with SciPy (CSR / BSR matrices)  
- 🧠 Conjugate Gradient (CG) solver integration  
- 🎲 Gaussian random field sampling  
- 📦 Modular forward model construction (`H` operator)  
- 📊 Designed for large structured datasets (maps / grids / fields)  
- 🔬 Scientific computing oriented (CMB / tomography / inverse problems)

---

## Mathematical Model

### Likelihood

\[
d = Hx + \epsilon, \quad \epsilon \sim \mathcal{N}(0, W^{-1})
\]

### Prior

\[
x \sim \mathcal{N}(\mu, C_x)
\]

### Posterior Sampling Step

Each Gibbs iteration solves:

\[
(H^T W H + C_x^{-1}) x = H^T W d + \eta
\]

where \(\eta\) represents stochastic Gaussian contributions ensuring correct posterior sampling.

---

## Algorithm Summary

At each iteration:

1. Sample Gaussian noise vectors  
2. Construct stochastic right-hand side:
   \[
   b = H^T W d + \text{noise terms}
   \]
3. Solve linear system using CG:
   \[
   (H^T W H + C_x^{-1}) x = b
   \]
4. Store the solution as a posterior sample

---

## Project Structure

```text
.
├── main.py                  # Main pipeline (data → model → sampling)
├── src/
│   ├── lib/                # Utilities (IO, templates, helpers)
│   ├── data/               # Data loading (Planck maps, simulations)
│   ├── models/             # Forward models (H operators)
│   ├── sampling/           # Gibbs / CG samplers
│   └── utils/              # Helper functions
├── data/                   # Input data (not tracked)
├── results/                # Output samples and diagnostics
└── README.md