# Approximate Inversion of Generalized Radon Transforms in Seismic Imaging

This repository contains Python code to simulate numerical experiments for the **approximate inversion** of **generalized Radon transforms (GRTs)** in **seismic imaging** using the imaging operator $\Lambda = K F^* \psi F$ as discussed in Ganster et al. (2023), *Approximate Inversion of a Class of Generalized Radon Transforms*. We extend their framework to handle laterally changing background velocities in 2D seismic imaging problems. 

The work builds on microlocal analysis, pseudo-differential/Fourier integral operator theory, and the framework of approximate inverses, and is applied to settings in seismic imaging.

This code accompanies my Master's thesis for the MSc in Mathematical Modelling and Scientific Computing at the University of Oxford and was used to generate the numerical results presented there.

---

## Overview

The approximate inversion approach implemented here:
- Handles general background velocity models.
- Supports 2D settings.
- Includes fast marching methods for solving the eikonal equation.
- Implements high-order fast sweeping schemes and a first-order finite element method for the transport equations.
- Uses the common offset acquisition geometry, but can easily be adjusted to other settings.


The methods are based on:
- **Microlocal analysis** for characterizing $\Lambda$.
- **Pseudo-differential and Fourier integral operators** for analyzing the GRT $F$.
- **Approximate inverse theory** for stable inversion.
- **Efficient numerical solvers** for PDEs arising in seismic imaging (both finite difference and finite element methods).

For the underlying theory, see:
- Ganster, K., & Rieder, A. (2023). Approximate Inversion of a Class of Generalized Radon Transforms. SIAM Journal on Imaging Sciences, 16(2), 842–866. https://doi.org/10.1137/22M1512417
- Louis, A. K. (1996). Approximate inverse for linear and some nonlinear problems. Inverse Problems, 12(2), 175–190. https://doi.org/10.1088/0266-5611/12/2/005
- Hörmander, L. (1971 - 2003). The analysis of linear partial differential operators I - IV. Springer.


---

## Requirements

- Python ≥ 3.9 (code written using Python 3.11.13)
- numpy
- scipy
- matplotlib
- scikit-fem
- scikit-image
- joblib
- eikonalfm


---

## Installation

You need Python ≥ 3.9 and `pip`.

```bash
git clone https://github.com/rauv-git/grt-approximate-inversion.git
cd grt-approximate-inversion
```

For the fast marching method solver:
```bash
pip install git+https://github.com/kevinganster/eikonalfm.git
```

For the FEM, parallelization, and contour finding:
```bash
pip install scikit-fem, joblib, scikit-image
```

You might want to use a virtual environment.

---

## Usage

Examples for the use are described in the Jupyter notebooks "general_example.ipynb" and "translation_invariant_example.ipynb". Whilst the former shows an example for a bilinear background velocity, the latter can be used to approximately invert the GRT for layered background velocities. Note that much longer runtimes are required in the general setting when compared to the layered setting.

---

## Features

- **Velocity Models**
  - Constant, layered, or laterally changing background velocities
- **Numerical Solvers**
  - Fast marching for eikonal equation
  - High-order fast sweeping and first-order FEM for transport equation

---

## Project Structure

```
grt-approximate-inversion/
├── grt_inversion/
│   ├── core/                         # Core interfaces for solvers and velocities
│   ├── reconstruction/               # Reconstruction algorithms
│   │   ├── general/                  # General background velocities
│   │   └── translation_invariant/    # Layered background velocities
│   ├── solvers/                      # PDE solvers
│   └── utils/                        # Utilities and visualization
├── general_example.ipynb
├── translation_invariant_example.ipynb
└── README.md
```
---

## Contributing

Pull requests and issues are welcome!  
If you plan a larger contribution, please first open an issue to discuss ideas.

---

## Citation

If you use this code in academic work, please cite:

```bibtex
@mastersthesis{rau2025approximateinversion,
  title={Approximate Inversion of Generalized Radon Transforms in Seismic Imaging},
  author={Rau, Vincent},
  school={University of Oxford},
  year={2025},
  type={{MSc} Thesis},
  note={MSc in Mathematical Modelling and Scientific Computing}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Finite difference solvers based on Ganster et al. (2023), Sethian (1999) and Zhang et al. (2006). General framework based on Ganster et al. (2023).
