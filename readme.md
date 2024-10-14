# Physics-Guided Self-Supervised Learning: Demonstration for Generalized RF Pulse Design

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  * [Installation](#Installation)
  * [Offline Training](#offline-training)
  * [Online Adaptation](#online-adaptation)
- [Publication](#publication)
- [Contacts](#contacts)
<!-- - [Star History](#star-history)-->

## Introduction
Generalized RF pulse design using Physics-guided Self-supervised learning (GPS) is a comprehensive framework for creating MRI RF pulses using a physics-guided, self-supervised learning approach. This algorithm can generate RF pulses that meet flexible design requirements by utilizing a physics module, the Bloch equation, to guide the learning and optimization process. The pulse design for 1D selective pulse, B1-insensitive pulse, SPatial-SPectral (SPSP) pulse, and 2D pulse is demonstrated. Through online adaptation, GPS can further compensate for MRI system imperfections, such as field inhomogeneity. For more details, see our [[paper]](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30307) published on Magnetic Resonance in Medicine.

![figure1.svg](resources%2Ffigure1.svg)


## Getting Started
The computing environment we tested.
- CPU: Intel Xeon Gold 6338 @ 2.00GHz
- GPU: NVIDIA A100

### Installation
0. Download and Install the appropriate version of NVIDIA driver and CUDA for your GPU.
1. Download and install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).
2. Clone this repo and cd to the project path.
```bash
git clone git@github.com:I3Tlab/GPS_RF.git
cd GPS_RF
```
3. Create and activate the Conda environment:
```bash
conda create --name GPSRF python=3.10.12
conda activate GPSRF
```
4. Install dependencies
```bash
pip install -r requirements.txt
```

### Offline training
Offline training indicates RF pulse design with homogenous fields.

#### 1D selective RF pulse (Figure 2a in the paper)
```bash
python 1D_pulse_demo.py
```

#### 1D B1-insensitive RF pulse (Figure 2b in the paper)
```bash
python 1D_adiabatic_demo.py
```

#### SPSP RF pulse (Figure 4 in the paper)

Fat suppression (Figure 4a in the paper)
```bash
python 1D_SPSP_demo.py --data_path data_loader/SPSP_TBW_3_SBW_6_pw_23p8ms_exc_width_5mm_water_192x96_conj.mat --notes water
```

Fat saturation (Figure 4b in the paper)
```bash
python 1D_SPSP_demo.py --data_path data_loader/SPSP_TBW_3_SBW_6_pw_23p8ms_exc_width_5mm_fat_192x96_conj.mat --notes fat
```

#### 2D RF pulse (Figure 5 in the paper)
```bash
python 2D_AI_demo.py
```

### Online Adaptation
Online adaptation indicates field inhomogeneity compensation by adjusting the RF pulse.
#### online adaptation for phantom scan (Figure 7 in the paper)
```bash
python 2D_online_adaptation_demo.py --B0 data_loader/measured_B0_20240407_3_phantom.mat --B1 data_loader/measured_B1_20240407_phantom.mat --notes phantom
```

#### online adaptation for phantom invivo brain scan (Figure 8 in the paper)
```bash
python 2D_online_adaptation_demo.py --B0 data_loader/measured_B0_20240415_2_brain.mat --B1 data_loader/measured_B1_20240415_brain.mat --notes invivo_brain
```

### Publication
```bibtex
@article{https://doi.org/10.1002/mrm.30307,
author = {Jang, Albert and He, Xingxin and Liu, Fang},
title = {Physics-guided self-supervised learning: Demonstration for generalized RF pulse design},
journal = {Magnetic Resonance in Medicine},
year = {2024}
volume = {early access},
number = {early access},
pages = {1-16},
keywords = {Bloch equations, deep learning, GPS, online adaptation, RF pulse, self-supervised learning},
doi = {https://doi.org/10.1002/mrm.30307},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.30307},
}
```

### Contacts
[Intelligent Imaging Innovation and Translation Lab](https://liulab.mgh.harvard.edu/) [[github]](https://github.com/I3Tlab) at the Athinoula A. Martinos Center of Massachusetts General Hospital and Harvard Medical School
* Albert Jang (awjang@mgh.harvard.edu)
* Xingxin He (xihe2@mgh.harvard.edu)
* Fang Liu (fliu12@mgh.harvard.edu)

149 13th Street, Suite 2301
Charlestown, Massachusetts 02129, USA
