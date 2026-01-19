![Docs](https://img.shields.io/github/actions/workflow/status/maximilian-kruse/SPIN/docs.yaml?label=Docs)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fmaximilian-kruse%2FSPIN%2Fmain%2Fpyproject.toml)
![License](https://img.shields.io/github/license/maximilian-kruse/SPIN)
![Beartype](https://github.com/beartype/beartype-assets/raw/main/badge/bear-ified.svg)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

# SPIN: Stochastic Process INference

SPIN is a Python package for the non-parametric Bayesian inference of the parameter functions of autonomous diffusion processes.
It can be used to infer the drift function $\mathbf{b}: \Omega\to\mathbb{R}^d$ and (squared) diffusion function $\mathbf{\Sigma}:\Omega\to\mathbb{R}^{d\times d}$ from trajectory data of an underlying process $\mathbf{X}$ on a domain $\Omega\subset\mathbb{R}^d$, indexed over $t\in\mathbb{R} _+ $,

$$
    d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t) dt + \sqrt{\mathbf{\Sigma}(\mathbf{X}_t)} d\mathbf{W}_t,\quad \mathbf{X}(t=0)=\mathbf{X}_0\ a.s.
$$

Under the hood, SPIN employs a PDE-based inference method, based on the **Kolmogorov equations** governing
the stochastic process under consideration. For more information on the underlying theory, we refer to the
**accompanying publication**,

>**[Non-parametric Inference for Diffusion Processes:
A Computational Approach via Bayesian Inversion
for PDEs](https://arxiv.org/abs/2411.02324)**

### Key Features
- **Non-parametric inference of drift and diffusion functions**
- **Works with stationary and time-dependent trajectory data**
- **PDE computations based on the finite element method**
- **Evaluation of the maximum a-posteriori estimate and Laplace approximation**
- **Generic and robust implementation, based on [hIPPYlib](https://dl.acm.org/doi/10.1145/3428447) and [FEniCS](https://fenicsproject.org/)**

## Getting Started

SPIN depends on a mixture of pip and conda dependencies, which can be efficiently managed using [Pixi](https://pixi.sh/latest/). To set up a virtual environment in which SPIN can be run, simply execute in the project root directory:

```bash
pixi install
```

The [documentation](https://maximilian-kruse.github.io/SPIN/) provides further information regarding usage, theoretical background, technical setup and API. We also provide runnable notebooks under [`examples`](https://github.com/maximilian-kruse/SPIN/tree/main/examples).
