# Fast Online Adaptive Neural MPC via Meta-Learning

[![arXiv](https://img.shields.io/badge/arXiv-2504.16369-brown)](https://arxiv.org/abs/2504.16369)
[![YouTube](https://img.shields.io/badge/Youtube-🎬-red)](https://www.youtube.com/watch?v=4K2QeBxWcWA)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)]()

This is the official implementation of the paper:  
[**"Fast Online Adaptive Neural MPC via Meta-Learning"**](https://arxiv.org/abs/2504.16369)  
by **Yu Mei**, **Xinyu Zhou**, **Shuyang Yu**, **Vaibhav Srivastava**, and **Xiaobo Tan**.

---

## 📑 Table of Contents
- [Fast Online Adaptive Neural MPC via Meta-Learning](#fast-online-adaptive-neural-mpc-via-meta-learning)
  - [📑 Table of Contents](#-table-of-contents)
  - [🎥 Demonstration Video](#-demonstration-video)
  - [🛠️ Installation Instructions](#️-installation-instructions)
    - [Option A: CUDA 12.x GPUs (e.g., RTX A2000)](#option-a-cuda-12x-gpus-eg-rtx-a2000)
    - [Option B: Blackwell GPUs (CUDA 13.x, e.g., RTX 5070)](#option-b-blackwell-gpus-cuda-13x-eg-rtx-5070)
  - [🚀 How to Run](#-how-to-run)
    - [1. Van-der-Pol Oscillator](#1-van-der-pol-oscillator)
    - [2. CartPole](#2-cartpole)
    - [3. 2D Quadrotor Stabilization and Tracking](#3-2d-quadrotor-stabilization-and-tracking)
      - [Stabilization Results](#stabilization-results)
      - [Tracking Results](#tracking-results)
  - [📚 Project Structure](#-project-structure)
  - [📝 Citation](#-citation)
---

## 🎥 Demonstration Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=4K2QeBxWcWA">
    <img src="http://img.youtube.com/vi/4K2QeBxWcWA/0.jpg" alt="Watch the video" width="60%" style="box-shadow: 1px 1px 6px rgba(0, 0, 0, 0.3); border-radius: 8px;">
  </a>
</p>

Watch our [**YouTube video**](https://www.youtube.com/watch?v=4K2QeBxWcWA) showcasing the control performance on the CartPole and 2D Quadrotor environments using the proposed Fast Online Meta-MPC framework.

---

## 🛠️ Installation Instructions

**1. Clone the repository** *(common to both tracks)*

   ```bash
   git clone https://github.com/yu-mei/MetaResidual-MPC.git
   cd MetaResidual-MPC
   ```

**2. Pick the track that matches your GPU**, then follow **one** of the two options below:

   | Your GPU | Track |
   |----------|-------|
   | Pre-Blackwell cards — RTX 20/30/40 series, A-series (e.g., **RTX A2000**) | [Option A](#option-a-cuda-12x-gpus-eg-rtx-a2000) |
   | Blackwell cards — RTX 50 series (e.g., **RTX 5070**), compute capability `sm_120` | [Option B](#option-b-blackwell-gpus-cuda-13x-eg-rtx-5070) |

---

### Option A: CUDA 12.x GPUs (e.g., RTX A2000)

**A1. Create a conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate l4control
   ```

**A2. Install `l4casadi`**

   Install the latest version using pip with `--no-build-isolation` (GPU/CUDA supported):

   ```bash
   pip install l4casadi --no-build-isolation
   ```

   > 🔗 Source: [github.com/Tim-Salzmann/l4casadi](https://github.com/Tim-Salzmann/l4casadi)

**A3. Install acados and the acados Python interface**

   A3.1 Clone and build Acados

   Follow the [official Acados installation guide](https://docs.acados.org/installation/index.html).

   A3.2 Install the Acados Python interface

   Follow the [Python interface installation guide](https://docs.acados.org/python_interface/index.html).

**A4. Install `safe-control-gym`**

   Follow the [official safe-control-gym installation guide](https://github.com/utiasDSL/safe-control-gym).

**A5. Override PyTorch installation**

   Due to version conflicts between `l4casadi` and `safe-control-gym`, it is necessary to override PyTorch:

   ```bash
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

   > ⚠️ This ensures compatibility with both `l4casadi` and `safe-control-gym`.  
   > 🔧 Make sure your CUDA drivers are compatible with CUDA 12.4.

**A6. Fix installation issues (if any)**

   If you encounter any remaining errors, manually install the missing or incompatible packages.  
   Package versions may vary depending on your system environment.

---

### Option B: Blackwell GPUs (CUDA 13.x, e.g., RTX 5070)

> ⚠️ **Why a separate track?** Blackwell cards (RTX 50 series) have compute capability `sm_120`, which PyTorch only supports from **2.7+** with **CUDA ≥ 12.8** builds. The Option A install (`pytorch==2.5.1 + pytorch-cuda=12.4`) imports fine but fails at the first GPU op with `CUDA capability sm_120 is not compatible with the current PyTorch installation`. In addition, the `-c pytorch` conda channel no longer publishes new versions, so PyTorch is installed via pip wheels here, and `l4casadi` moves to the **last** step so that it compiles against the final PyTorch.

> 💡 The `CUDA Version: 13.x` shown by `nvidia-smi` is the *driver's* maximum supported runtime, not an installed toolkit. The pip wheels bundle their own CUDA runtime, so no system CUDA toolkit is required for PyTorch.

**B1. Create a conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate l4control
   ```

**B2. Install acados and the acados Python interface** *(same as A3 — unchanged)*

   The acados C library is CPU-only and unaffected by the GPU swap.

   B2.1 Clone and build Acados

   Follow the [official Acados installation guide](https://docs.acados.org/installation/index.html).

   B2.2 Install the Acados Python interface

   Follow the [Python interface installation guide](https://docs.acados.org/python_interface/index.html).

**B3. Install `safe-control-gym`** *(same as A4)*

   Follow the [official safe-control-gym installation guide](https://github.com/utiasDSL/safe-control-gym).

   > 💡 Whatever PyTorch version it pulls in will be replaced in the next step — ignore it for now.

**B4. Install PyTorch (Blackwell build)** *(replaces A5 — do **not** run the old conda command)*

   ```bash
   pip uninstall -y torch torchvision torchaudio triton
   pip install "torch==2.9.*" "torchvision==0.24.*" "torchaudio==2.9.*" \
       --index-url https://download.pytorch.org/whl/cu130 --no-cache-dir
   ```

   > 💡 torch ≥ 2.7 is the hard minimum for `sm_120`; the 2.9 series is the first with CUDA 13.0 wheels, still supports Python 3.10, and is a modest API jump from 2.5.1.

   Conservative fallback if the code misbehaves under 2.9 (cu128 wheels remain hosted and run fine on 13.x drivers):

   ```bash
   pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
       --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
   ```

   > 🔗 Source: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) · pinned versions: [pytorch.org/get-started/previous-versions](https://pytorch.org/get-started/previous-versions/)

**B5. Install `l4casadi` — last, after PyTorch** *(moved from A2)*

   ```bash
   pip install l4casadi --no-build-isolation --no-cache-dir
   ```

   > ⚠️ Order matters: `--no-build-isolation` makes `l4casadi` compile and link against the *currently installed* PyTorch, so it must come **after** B4. If you ever change the torch version later, force a rebuild:
   > `pip install l4casadi --no-build-isolation --force-reinstall --no-cache-dir`  
   > 🔧 Requires GCC ≥ 10. If the GPU build is not detected automatically and a system toolkit exists: `CUDACXX=/usr/local/cuda/bin/nvcc pip install l4casadi --no-build-isolation`.  
   > 🔗 Source: [github.com/Tim-Salzmann/l4casadi](https://github.com/Tim-Salzmann/l4casadi)

**B6. Verify the GPU stack**

   ```python
   import torch
   print(torch.__version__, torch.version.cuda)   # e.g. 2.9.x  13.0
   print(torch.cuda.get_device_name(0))           # NVIDIA GeForce RTX 5070
   print(torch.cuda.get_device_capability(0))     # (12, 0)
   x = torch.randn(1024, 1024, device="cuda")
   print((x @ x).sum())                           # real kernel launch = the actual test
   ```

   > ⚠️ `torch.cuda.is_available()` only checks the driver — always run the matmul line. If you see `sm_120 is not compatible` or `no kernel image is available`, a pre-cu128 or CPU wheel sneaked in: redo B4 (uninstall first, keep `--no-cache-dir` and the explicit `--index-url`).

**B7. Fix installation issues (if any)**

   If you encounter any remaining errors, manually install the missing or incompatible packages.  
   Package versions may vary depending on your system environment.

   > 💡 If you reinstall `safe-control-gym` later, use `pip install -e . --no-deps` so it does not downgrade PyTorch.  
   > 💡 If your local copy of `environment.yml` predates the repo cleanup and still lists `nvidia-*-cu11` or `triton==3.1.0` under `pip:`, delete those lines (or `pip uninstall` the packages) — they are leftovers from an old CUDA 11-era PyTorch and conflict with the Blackwell wheels.

---

## 🚀 How to Run

### 1. Van-der-Pol Oscillator
We provide several scripts under `VanderPol/` for running different versions of the Van der Pol system using Meta-MPC:

| Script | Description |
|--------|-------------|
| `VanderPolSys_sim.py` | Simulates the nominal Van der Pol system. |
| `VanderPolSys_real.py` | Simulates the real system (with mismatched dynamics). |
| `VanderPolSys_naive.py` | Runs nominal MPC to predict the trajectories. |
| `VanderPolSys_naive_lightmlp.py` | Runs nominal MPC with a lightweight learned MLP model (learn from stratch)|
| `VanderPolSys_naive_meta.py` | Runs MPC using a meta-learned model. |
| `VanderPolSys_Collection_Meta.py` | Collects offline data for training the meta-learned model, and data file `vdp_meta_nominal_residual.py` is under dataset|
| `Comparsion.ipynb` | Jupyter notebook comparing performance across methods. |
| `MetaLearning/Offline_Train_Meta.py`| Training the Meta MLP offline using the data file `vdp_meta_nominal_residual.py`|

📌 Example: Run nominal MPC + Meta MLP

```bash
cd VanderPol
python VanderPolsys_naive_meta.py
```

After running the different methods and saving results in the `results/` folder, open `Comparsion.ipynb` to visualize and compare the performance.

<p align="center">
  <img src="assets/vanderpol_results.png" alt="Van der Pol Results" width="75%">
</p>

---

### 2. CartPole

We provide several scripts under `Cartpole/` to run different MPC controllers for the CartPole system using our Meta-MPC framework:

| Script | Description |
|--------|-------------|
| `cartpole_Nominal.py` | Runs MPC with a nominal (physics-based) model. |
| `cartpole_LightMLP.py` | Runs MPC using a learned residual MLP trained from scratch. |
| `cartpole_MetaMLP.py` | Runs MPC using a meta-learned residual MLP with online adaptation. |
| `cartpole_Nominal_seeds.py` | Batch test across seeds using nominal model. |
| `cartpole_LightMLP_seeds.py` | Batch test using residual MLP (non-meta). |
| `cartpole_MetaMLP_seeds.py` | Batch test using meta-residual MLP model. |
| `MetaLearning/DataCollection_Meta.py` | Collects residual training data for meta-learning. The output is saved in `meta_dataset_mpc/`. |
| `MetaLearning/Offline_Train_Meta.py` | Trains the Meta-Residual MLP model using the collected CSV dataset. |
| `meta_dataset_mpc/cartpole_meta_residual_mpc.csv` | CSV dataset collected from `DataCollection_Meta.py` used for offline meta-learning. |
| `Comparsion.ipynb` | Jupyter notebook to visualize and compare results across all methods. |

📌 Example: Run Meta-MPC with Online Adaptation

```bash
cd Cartpole
python cartpole_MetaMLP.py
```

📌 Example: Collect Residual Data for Meta-Training

```bash
python MetaLearning/DataCollection_Meta.py
```

📌 Example: Train Meta Residual MLP Offline

```bash
python MetaLearning/Offline_Train_Meta.py
```

After running all variants, results will be saved in the `results/` folder.  
Open `Comparsion.ipynb` to visualize metrics such as RMSE, trajectory tracking, and adaptation efficiency.

<p align="center">
  <img src="assets/CartPole_Nominal.gif" alt="Nominal MPC" width="30%" style="margin-right: 10px;">
  <img src="assets/CartPole_MLP.gif" alt="Neural MPC + MLP" width="30%" style="margin-right: 10px;">
  <img src="assets/CartPole_MetaMLP.gif" alt="Neural MPC + MetaMLP" width="30%">
</p>

<p align="center">
  <em>Left: Nominal MPC &nbsp; | &nbsp; Middle: Neural MPC + Residual MLP &nbsp; | &nbsp; Right: Neural MPC + Residual Meta-MLP</em>
</p>

---

### 3. 2D Quadrotor Stabilization and Tracking

We provide two folders for 2D Quadrotor control tasks using our Meta-MPC framework:

- `Quadrotor_2D_Stabilization/`: for stabilization tasks
- `Quadrotor_2D_Tracking/`: for reference trajectory tracking

Each folder contains scripts to run, which is similar as CartPole system:

📌 Example: Run Meta-MPC for Stabilization

```bash
cd Quadrotor_2D_Stabilization
python quadrotor2D_Meta.py
```

📌 Example: Run Meta-MPC for Tracking

```bash
cd Quadrotor_2D_Tracking
python quadrotor2D_Meta.py
```

#### Stabilization Results

<p align="center">
  <img src="assets/Quadrotor_stabilization_Nominal.gif" alt="Nominal MPC" width="32%" style="margin-right: 8px;">
  <img src="assets/Quadrotor_stabilization_MLP.gif" alt="MLP Residual MPC" width="32%" style="margin-right: 8px;">
  <img src="assets/Quadrotor_stabilization_MetaMLP.gif" alt="Meta-Residual MPC" width="32%">
</p>

<p align="center">
  <em>Left: Nominal MPC &nbsp; | &nbsp; Middle: MPC + Residual MLP &nbsp; | &nbsp; Right: Neural MPC + Residual Meta-MLP</em>
</p>

---

#### Tracking Results

<p align="center">
  <img src="assets/Quadrotor_tracking_Nominal.gif" alt="Nominal MPC" width="32%" style="margin-right: 8px;">
  <img src="assets/Quadrotor_tracking_MLP.gif" alt="MLP Residual MPC" width="32%" style="margin-right: 8px;">
  <img src="assets/Quadrotor_tracking_MetaMLP.gif" alt="Meta-Residual MPC" width="32%">
</p>

<p align="center">
  <em>Left: Nominal MPC &nbsp; | &nbsp; Middle: MPC + Residual MLP &nbsp; | &nbsp; Right: Neural MPC + Residual Meta-MLP</em>
</p>

---

## 📚 Project Structure

```
MetaResidual-MPC/
├── assets/                      # Demo GIFs and figures used in README (e.g., CartPole_MetaMLP.gif, Quadrotor_*.gif)
├── Cartpole/                    # Code for CartPole experiments
├── Quadrotor_2D_Stabilization/  # Code for 2D Quadrotor stabilization tasks
├── Quadrotor_2D_Tracking/       # Code for 2D Quadrotor trajectory tracking tasks
├── VanderPol/                   # Code for Van der Pol oscillator experiments
├── environment.yml              # Conda environment file
└── README.md                    # Project documentation
```

---

## 📝 Citation

If you find our work useful, please consider citing:

```bibtex
@article{mei2025fast,
  title={Fast Online Adaptive Neural MPC via Meta-Learning},
  author={Mei, Yu and Zhou, Xinyu and Yu, Shuyang and Srivastava, Vaibhav and Tan, Xiaobo},
  journal={IFAC-PapersOnLine},
  volume={59},
  number={30},
  pages={377--382},
  year={2025},
  publisher={Elsevier}
}
```

---
