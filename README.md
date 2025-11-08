
<h1 align="center">🌀 Multi-Scale Network with Provable Lipschitz Continuity for Universal CT Reconstruction</h1>

<p align="center">
  <strong>Baoshun Shi, Xinya Ji, Ke Jiang, Huazhu Fu</strong><br>
  <em>PR 2025 (Under Review)</em>
</p>


<p align="center">
  <a href="https://github.com/shibaoshun/PromptCT">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="github">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c?logo=pytorch" alt="pytorch">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="license">
</p>


---

> **Abstract**  
> Although few-view computed tomography (CT), including sparse-view CT (SVCT) and limited-angle CT (LACT), reduces radiation dose, reconstructed CT images often suffer from severe artifacts due to incomplete projection data. Deep learning-based (DL-based) reconstruction methods can eliminate some artifacts, but they still face three main limitations: (i) The interpretability and provability of prior networks in deep unfolding methods often remain challenging, as these networks are empirically designed. (ii) A solution to the aforementioned limitation is to use sparse representation model-driven networks, but these architectures often ignore multi-scale information. (iii) Existing DL-based methods handle varying view numbers separately, resulting in high storage costs. To address these limitations, we propose a prompting multi-scale sparsifying frame network, dubbed Proms-Net. Proms-Net embraces cross-scale complementarity to leverage features across different scales. Furthermore, by embedding Proms-Net as a prior network, we develop a PRompting Iterative Optimization network for universal CT Reconstruction (**PRIOR**), which adapts to multiple sampling views using a single model. Experiments demonstrate that PRIOR achieves outstanding performance with a unified model using a single set of pretrained weights across varying view numbers, reducing storage costs in both SVCT and LACT reconstruction tasks. On the theoretical side, we explicitly prove that Proms-Net satisfies Lipschitz continuity and further show the convergence of the corresponding iterative algorithm.
---
**Network Architecture:**  
![Network Architecture](https://github.com/shibaoshun/PRIOR/blob/main/fig/Proms-Net.jpg?raw=true)
![Network Architecture](https://github.com/shibaoshun/PRIOR/blob/main/fig/PRIOR.jpg?raw=true)

**Reconstruction Results:**  
![Reconstruction Results](https://github.com/shibaoshun/PRIOR/blob/main/fig/SVCT.jpg?raw=true)
![Reconstruction Results](https://github.com/shibaoshun/PRIOR/blob/main/fig/LACT.jpg?raw=true)




## 📚 Table of Contents

- [🚀 Installation](#-installation)
- [📂 Dataset Preparation](#-dataset-preparation)
- [🧠 Training](#-training)
- [🔍 Testing](#-testing)
- [📈 Results](#-results)
- [📄 Citation](#-citation)
- [📜 License & Acknowledgement](#-license--acknowledgement)

---

## 🚀 Installation

This project is implemented with **PyTorch 1.10.0** and supports both **CPU** and **GPU (CUDA)** environments.  
For best performance, we recommend running on a GPU-enabled system.

### ✅ Recommended Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.10.0  
- CUDA ≥ 11.3  
- GPU: NVIDIA RTX 3090 / A100 or equivalent  
- OS: Ubuntu 20.04 / Windows 10+

### 🔧 Installation Steps

### Please make sure your environment meets the following requirements:

```bash
git clone https://github.com/shibaoshun/PRIOR.git
cd PRIOR
pip install -r requirements.txt
```

## 📂 Dataset Preparation

You can download the **training and testing datasets** from Baidu Drive:


📁 **Dataset**

 🔗 [https://pan.baidu.com/s/1lQRFUrkaUH7uEDB6iyKq0Q?pwd=2025](https://pan.baidu.com/s/16bQk82x7qzOfViV71hNS9A?pwd=2025)
 
 🔑 **Password:** `2025`

After downloading, place the data under:

```
PRIOR/
 ├── data/
 │    ├── train/
 │    ├── val/
 │    └── test/
```

------

## 🧠 Training

You can train the model using the default configuration with a single command:

```
python main.py --phase tr
```

------

## 🔍 Testing

### 🔸 Pretrained Models

Download pretrained models from:




📁 **Pretrained Checkpoints**

 🔗 [https://pan.quark.cn/s/c828b2da6684](https://pan.quark.cn/s/c828b2da6684)
 
 🔑 **Password:** `Ebkn`

Place them under:

```
PRIOR/
 ├── result/
 │    ├── sparse/
 │    │    ├── ckp/
 │    │    │    └── grad_best.pth
```

### 🔸 Example Testing Command

```
python main.py --phase test
```

> Results and reconstruction outputs will be saved under `./result/`.

------

## 📈 Results

PromptCT achieves **superior reconstruction quality** with significantly **lower storage costs**, enabling **multiple sparse-view CT reconstruction in a single model**.

| Method    | PSNR (dB) ↑ | SSIM ↑ | RMSE ↑ | Storage ↓  |
|:----------|:-----------:|--------|:------:|:----------:|
| FBP       |    16.17    | 0.5719 | 0.1655 |     -      |
| PromptIR  |    39.16    | 0.9407 | 0.0117 |   337.8    |
| AMIR      |    37.43    | 0.9230 | 0.0146 |   269.9    |
| CAPTNet   |    37.94    | 0.9306 | 0.0155 |   286.1    |
| PRIOR-B   |    39.46    | 0.9440 | 0.0110 |    80.2    |
| **PRIOR** |    40.06    | 0.9473 | 0.0103 |    83.3    |

------

## 📜 License & Acknowledgement

This project is released under the MIT License.
 We would like to thank all contributors and referenced works for their valuable resources and datasets.

------

<p align="center">⭐ If you find this repository useful, please give it a star to support us! ⭐</p> ```

