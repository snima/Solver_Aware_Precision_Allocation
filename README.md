# Storage Format Allocation and Arithmetic Precision in Robust Matrix Completion

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![Build Status](https://img.shields.io/badge/Tests-9%20Passing-success.svg)](tests/)
[![Dependencies](https://img.shields.io/badge/Dependencies-Pure%20NumPy%20%2F%20SciPy-informational.svg)](requirements.txt)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-100%25%20Verified-blueviolet.svg)](reproduce_all.py)

Official repository for the research paper:  
**"Storage Format Allocation and Arithmetic Precision in Robust Matrix Completion"**  
*Nima Sahraneshinsamani, José I. Aliaga, Sandra Catalán, and José R. Herrero*  
*(CMMSE / Journal of Computational and Applied Mathematics)*

---

## 🌟 Interactive Visual Simulator

Try the live interactive web simulator hosted directly on GitHub Pages:
👉 **[Open Interactive Demonstration](https://snima.github.io/Solver_Aware_Precision_Allocation/)** *(or open [`docs/index.html`](docs/index.html) locally in any browser)*

Explore real-time Stiefel manifold factor trajectories, Cauchy outlier suppression weights, and side-by-side comparisons of uniform vs. influence-routed mixed precision.

---

## 📖 Key Research Insights

Modern hardware accelerators (such as NVIDIA Blackwell and AMD Instinct) increasingly adopt native 8-bit (FP8) and 4-bit (FP4) formats to combat the memory-bandwidth wall. However, blindly applying uniform 4-bit storage to continuous physical sensor observations triggers severe degradation (up to **61.4-fold error explosion** on synthetic problems and severe breakdown on real-world traffic networks).

This work establishes three core principles:
1. **Decoupled Precision Architecture**: We separate *Observation Storage Precision* from *Solver Arithmetic Precision*. Truncating solver arithmetic is numerically fragile (FP8 arithmetic multiplies reconstruction error by $9.13\times$), whereas observation storage is remarkably forgiving when guided selectively.
2. **Riemannian Stiefel IRLS**: We optimize $U$ on the compact Stiefel manifold $\mathrm{St}(m, r) = \{U \in \mathbb{R}^{m \times r} : U^\top U = I_r\}$. This anchors the factor scale gauge, eliminating the exponential drift of unconstrained alternating least squares ($1.121$ penalty vs $2.148$). The redescending Cauchy M-estimator dynamically downweights gross sensor spikes ($389.6\times$ suppression), preventing the scarce high-precision budget from being wasted on corruptions.
3. **Solver-Aware Allocation Rules**:
   - **Quantization-Error Allocation**: Minimizes the entrywise first-order Riemannian gradient perturbation bound; strictly dominates in static storage (FP8).
   - **Residual-Influence Allocation**: Prioritizes entries with high gradient leverage; stabilizes dynamic reallocation under ultra-coarse 4-bit formats (FP4).
   - **Column-Balanced Fairness Quotas**: Prevents numerical rank collapse in spatial sensor networks where individual stations have sparse observations, achieving up to **$56.6\%$ active storage reduction** relative to FP32 ($78.3\%$ relative to FP64) in highway traffic and up to **$74.9\%$** in climate grids with under $1\%$ reallocation overhead.

---

## 🚀 Quickstart (< 1 Minute)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/snima/Solver_Aware_Precision_Allocation.git
cd Solver_Aware_Precision_Allocation

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all quantizers, Riemannian solvers, and storage accounting modules pass:
```bash
python3 -m unittest discover -s tests -v
```

### 3. Run Benchmark Smoke Test (< 1s)
```bash
python3 reproduce_all.py --benchmark synthetic
```

---

## 📊 Master Reproducibility Command Matrix

Every table, figure, and empirical benchmark reported in the manuscript can be replicated with a single line:

| Target / Study | Description | Command | Runtime | Output |
| :--- | :--- | :--- | :--- | :--- |
| **Table 6 (Paper)** | Caltrans PeMS08 Highway Traffic (24h, 170 detectors $\times$ 288 timesteps) | `python3 reproduce_all.py --benchmark pems` | ~8s | Terminal Summary & Penalties |
| **Extended PeMS** | 2-Week Highway Continuous Flow (170 $\times$ 4,032) | `python3 benchmarks/pems_traffic/run_pems.py --mode 2weeks` | ~20s | Terminal Table |
| **Full PeMS08 Network** | Massive 62-Day Network (3.03 Million Spatiotemporal Entries) | `python3 benchmarks/pems_traffic/run_pems.py --mode full` | ~35s | Terminal Table |
| **Climate Sensing** | Continental US EPA PM2.5 Grid (710 stations $\times$ 365 days, 2023) | `python3 reproduce_all.py --benchmark climate --climate-seeds 1` | ~5s | `results/climate_campaign_results.csv` |
| **Multi-Seed Climate** | 5-Seed Monte Carlo Study with Wildfire Plume Transients | `python3 reproduce_all.py --benchmark climate --climate-seeds 5` | ~25s | Aggregated Statistical Table |
| **Synthetic Sweeps** | Systematic Format $\times$ Policy $\times$ Budget Grid | `python3 benchmarks/synthetic/run_synthetic.py --m 128 --n 128 --seeds 3` | ~15s | Parameter Grid Table |
| **All Benchmarks** | Sequentially executes all core benchmarks | `python3 reproduce_all.py --benchmark all` | ~30s | Complete Validation Log |

---

## 🏗️ Repository Architecture

```
github_release/
├── README.md                          # Master academic guide and documentation
├── LICENSE                            # Apache License 2.0 (open for research & commercial use)
├── requirements.txt                   # Minimal runtime dependencies (numpy, scipy, matplotlib)
├── CITATION.cff                       # Machine-readable citation metadata
├── CITATIONS.bib                      # BibTeX citation entry
├── reproduce_all.py                   # Master CLI entrypoint for all paper benchmarks
│
├── src/                               # Core mathematical engine
│   ├── __init__.py
│   └── mixed_precision_mc.py          # Stiefel Riemannian IRLS, Cauchy loss, routing & quantizers
│
├── benchmarks/                        # Real-world and synthetic benchmark suites
│   ├── synthetic/
│   │   └── run_synthetic.py           # Controlled low-rank generator and budget sweeper
│   ├── pems_traffic/                  # Caltrans PeMS08 highway network
│   │   ├── run_pems.py                # Table 6 reproduction & multi-week modes
│   │   └── data/PEMS08.npz            # 170 detectors x 17,856 timestamps
│   └── climate_epa/                   # US EPA nationwide PM2.5 continental grid (2023)
│       ├── run_climate.py             # Multi-seed climate campaign runner
│       └── data/epa_pm25_processed.npz# 710 stations x 365 days
│
├── tests/                             # Unit test suite
│   ├── __init__.py
│   └── test_mixed_precision_mc.py     # 9 comprehensive tests (quantization, convergence, Stiefel)
│
├── docs/                              # GitHub Pages site
│   └── index.html                     # Rich interactive visual simulator
│
└── results/                           # Output directory for exported CSV summaries
```

---

## 🔬 Mathematical Formulation

### 1. Optimization Problem
$$\min_{U \in \mathrm{St}(m,r),\, V \in \mathbb{R}^{n \times r}} \sum_{(i,j) \in \Omega} \rho\left(\frac{(U V^\top - Y)_{ij}}{\sigma}\right) + \frac{\lambda}{2} \|V\|_F^2$$
where:
- $\mathrm{St}(m, r) = \{U \in \mathbb{R}^{m \times r} : U^\top U = I_r\}$ is the compact Stiefel manifold.
- $\rho(z) = \frac{c^2}{2} \ln\left(1 + \frac{z^2}{c^2}\right)$ is the Cauchy M-estimator loss function.
- $w(z) = \frac{\rho'(z)}{z} = \frac{1}{1 + z^2 / c^2}$ are the redescending IRLS weights.
- $\sigma = 1.4826 \cdot \mathrm{median}(|r|)$ is the consistent Median Absolute Deviation (MAD) scale estimator.

### 2. Alternating Riemannian IRLS Steps
- **$V$-Step**: Independent weighted linear least-squares solve per column $j \in \{1,\dots,n\}$:
  $$(U_{\Omega_j}^\top W_j U_{\Omega_j} + \lambda I_r) v_j = U_{\Omega_j}^\top (W_j y_{\Omega_j})$$
- **$U$-Step**: Riemannian gradient descent with Armijo backtracking along polar retraction:
  - Tangent projection: $P_{T_U}(G) = G - U\,\mathrm{sym}(U^\top G)$ where $\mathrm{sym}(A) = \frac{A + A^\top}{2}$.
  - Polar retraction: $R_U(\xi) = \mathrm{polar}(U + \xi) = L R^\top$ where $U + \xi = L \Sigma R^\top$.

### 3. Precision Allocation Policies
Given an FP32 budget fraction $\beta \in [0, 1]$ (protecting $k = \lfloor \beta |\Omega| \rfloor$ entries):
- **Quantization-Error-Aware**: $\mathrm{Score}_{ij} = w_{ij} \cdot |Y_{ij} - Q(Y_{ij})| \cdot \|v_j\|_2$  
  *(Minimizes entrywise Riemannian gradient perturbation bound; dominant for static storage)*
- **Residual-Influence-Aware**: $\mathrm{Score}_{ij} = w_{ij} \cdot |r_{ij}| \cdot \|v_j\|_2$  
  *(First-order sensitivity of the objective gradient; stabilizes dynamic 4-bit reallocation)*
- **Column-Balanced Allocation**: Guarantees each column $j$ a minimum quota $q_j = \lceil \alpha |\Omega_j| \rceil$ of protected entries before allocating remaining budget globally, preventing column rank collapse in sensor networks.

---

## 💾 Hardware Memory Accounting

Active storage bit-rates account for true physical hardware overhead:
$$\text{Bits per Observation} = \frac{32 \cdot |H| + b_{\mathrm{low}} \cdot |\Omega \setminus H| + |\Omega| + 16 \cdot N_{\mathrm{blocks}}}{|\Omega|}$$
- $32 \cdot |H|$: 32 bits per entry in the protected FP32 pool.
- $b_{\mathrm{low}}$: Bit-width of coarse storage (e.g., 8 bits for FP8, 4 bits for FP4).
- $+1$ bit: Dedicated binary hardware routing mask per observation.
- $+16$ bits: Shared 16-bit block scale factor per block of 32 contiguous low-precision entries (compliant with Open Compute Project / NVFP microscaling specification).

---

## 🌐 Setting Up GitHub Pages (Interactive Demo)

To enable the live interactive demo on GitHub:
1. Push this repository to your GitHub account.
2. Go to **Settings** > **Pages**.
3. Under **Build and deployment** > **Branch**, select `main` (or your default branch) and choose the `/docs` folder.
4. Click **Save**. Within 1–2 minutes, your interactive demo will be live at `https://snima.github.io/Solver_Aware_Precision_Allocation/`!

---

## 📜 Citation

If you find this codebase or methodology useful in your research, please cite our paper:

```bibtex
@article{sahraneshinsamani2026storage,
  title   = {Storage Format Allocation and Arithmetic Precision in Robust Matrix Completion},
  author  = {Sahraneshinsamani, Nima and Aliaga, Jos{\'e} I. and Catal{\'a}n, Sandra and Herrero, Jos{\'e} R.},
  journal = {Journal of Computational and Applied Mathematics},
  year    = {2026},
  note    = {Under review / In press}
}
```

---

## 📄 License & Commercial Rights

This project is licensed under the **[Apache License 2.0](LICENSE)**. Open for academic, research, and commercial usage with explicit patent grant protection.

**Corresponding Author:** Nima Sahraneshinsamani (`sahrans@uji.es`), Universitat Jaume I, Spain.
