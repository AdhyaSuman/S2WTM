# S2WTM: Spherical Sliced-Wasserstein Autoencoder for Topic Modeling

-----

## 🔍 Framework Overview

<p align="center">
  <img src="https://github.com/AdhyaSuman/S2WTM/blob/main/Miscellaneous/S2WTM_framework.png" width="600"/>
</p>

-----

## 📊 Datasets

We used the following datasets for evaluation:

| Dataset | # Docs | # Words | # Labels |
|---|---|---|---|
| **20Newsgroups** (20NG) | 16,309 | 1,612 | 20 |
| **BBC News** (BBC) | 2,225 | 2,949 | 5 |
| **M10** | 8,355 | 1,696 | 10 |
| **SearchSnippets** (SS) | 12,295 | 2,000 | 8 |
| **Pascal** | 4,834 | 2,630 | 20 |
| **Biomedical** (Bio) | 19,448 | 2,000 | 20 |
| **DBLP** | 54,595 | 1,513 | 4 |

-----

## 📘 Tutorials

To understand and use S2WTM efficiently, we provide a tutorial notebook that demonstrates how to run the model, evaluate results, and explore the outputs.

You can open it directly in Google Colab using the badge below:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdhyaSuman/S2WTM/blob/main/Notebooks/Example.ipynb)

> 📁 **Path:** `Notebooks/Example.ipynb`

This notebook demonstrates:

  * How to load a dataset and configure S2WTM
  * Run the topic modeling pipeline

-----

## 📖 Citation

This work is available on ArXiv\! 📄

📄 Read the paper:

  - **[ArXiv](https://arxiv.org/abs/2507.12451)**

### 📌 BibTeX

```bibtex
@inproceedings{adhya-sanyal-2025-s2wtm,
    title = "{S}2{WTM}: {S}pherical {S}liced-{W}asserstein {A}utoencoder for {T}opic {M}odeling",
    author = "Adhya, Suman  and
      Sanyal, Debarshi Kumar",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1131/",
    doi = "10.18653/v1/2025.acl-long.1131",
    pages = "23211--23225",
    ISBN = "979-8-89176-251-0",
    abstract = "Modeling latent representations in a hyperspherical space has proven effective for capturing directional similarities in high-dimensional text data, benefiting topic modeling. Variational autoencoder-based neural topic models (VAE-NTMs) commonly adopt the von Mises-Fisher prior to encode hyperspherical structure. However, VAE-NTMs often suffer from posterior collapse, where the KL divergence term in the objective function highly diminishes, leading to ineffective latent representations. To mitigate this issue while modeling hyperspherical structure in the latent space, we propose the Spherical Sliced Wasserstein Autoencoder for Topic Modeling (S2WTM). S2WTM employs a prior distribution supported on the unit hypersphere and leverages the Spherical Sliced-Wasserstein distance to align the aggregated posterior distribution with the prior. Experimental results demonstrate that S2WTM outperforms state-of-the-art topic models, generating more coherent and diverse topics while improving performance on downstream tasks."
}
```

-----

## Acknowledgment

All experiments were conducted using **[OCTIS](https://github.com/MIND-Lab/OCTIS)**, an integrated framework for topic modeling, comparison, and optimization.

📌 **Reference:** Silvia Terragni, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. (2021). *"OCTIS: Comparing and Optimizing Topic Models is Simple\!"* [EACL](https://www.aclweb.org/anthology/2021.eacl-demos.31/).

-----

🌟 **If you find this work useful, please consider citing our paper and giving a star ⭐ to the repository\!**
