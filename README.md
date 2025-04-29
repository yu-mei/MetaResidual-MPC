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
  - [🔥 News](#-news)
  - [🎥 Demonstration Video](#-demonstration-video)
  - [🛠️ Installation Instructions](#️-installation-instructions)
  - [🚀 How to Run](#-how-to-run)
  - [📚 Project Structure](#-project-structure)
  - [📝 Citation](#-citation)
---

## 🔥 News
- \[2025-04\] Paper "**Fast Online Adaptive Neural MPC via Meta-Learning**" uploaded to [arXiv](https://arxiv.org/abs/2504.16369).
- \[2025-04\] Code for training, online adaptation, and MPC execution is released.

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

1. **Clone the repository**

   ```bash
   git clone https://github.com/yu-mei/MetaResidual-MPC.git
   cd MetaResidual-MPC
   ```

2. **Create a conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate l4control
   ```

3. **Install `l4casadi`**

   Install the latest version using pip with no-build-isolation (I used GPU(CUDA) in my case):

   ```bash
   pip install l4casadi --no-build-isolation
   ```

   > 🔗 Source: [github.com/Tim-Salzmann/l4casadi](https://github.com/Tim-Salzmann/l4casadi) 
   
4. **Install `acados` and the `acados` Python interface**

   4.1 **Clone and build Acados**

   Follow the [official Acados installation guide](https://docs.acados.org/installation/index.html):

   4.2 **Install the Acados Python interface**

   Follow the [Python interface installation guide](https://docs.acados.org/python_interface/index.html):

   4.3 **Set the `ACADOS_INSTALL_DIR` environment variable**

   After installation, set the environment variable to point to your Acados directory.

5. **Install safe-control-gym**
   Follow the [official safe-control-gym installation guide](https://github.com/utiasDSL/safe-control-gym)

6. **Install PyTorch (Override)**

   Due to version conflicts between `l4casadi` and `safe-control-gym`, it is necessary to override the PyTorch installation.

   Install PyTorch 2.5.1 with CUDA 12.4 support:

   ```bash
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

   > ⚠️ This step ensures compatibility with both `l4casadi` and `safe-control-gym`.  
   > Make sure your system's CUDA drivers are compatible with CUDA 12.4.

7. **Fix Installation**
    If you encounter any remaining installation errors, please manually install the missing or incompatible packages.  
   Note that the exact package versions may vary depending on your system environment.

---

## 🚀 How to Run


---

## 📚 Project Structure

```
MetaResidual-MPC/
├── Cartpole/                    # Code for CartPole experiments
├── Quadrotor_2D_Stabilization/   # Code for 2D Quadrotor stabilization tasks
├── Quadrotor_2D_Tracking/        # Code for 2D Quadrotor trajectory tracking tasks
├── VanderPol/                    # Code for Van der Pol oscillator experiments
├── environment.yml               # Conda environment file
└── README.md                     # Project documentation
```

---

## 📝 Citation

If you find our work useful, please consider citing:

```bibtex
@article{mei2025fast,
  title={Fast Online Adaptive Neural MPC via Meta-Learning},
  author={Mei, Yu and Zhou, Xinyu and Yu, Shuyang and Srivastava, Vaibhav and Tan, Xiaobo},
  journal={arXiv preprint arXiv:2504.16369},
  year={2025}
}
```

---

