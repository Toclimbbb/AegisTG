# 🛡️ AegisTG: Official Implementation (Anonymous Submission for ICIG 2026)

> **⚠️ DOUBLE-BLIND REVIEW DECLARATION**
> 
> This repository is specifically created to comply with the strict double-blind review policy of the International Conference on Image and Graphics (ICIG 2026). It serves exclusively as a supplementary material to facilitate the peer-review process. 
>
> All necessary core code snippets, model architectures, and inference pipelines have been carefully sanitized and provided herein for reviewers to evaluate the technical merit, reproducibility, and architectural innovations of our proposed **AegisTG** framework. 
> 
> **Intellectual Property & Full Release Statement:** Due to ongoing intellectual property protection and institutional policies, the current repository contains a curated subset of the codebase. Upon the official acceptance of the paper, the complete, end-to-end training framework, pre-trained weights, deployment scripts, and detailed documentation will be immediately and fully open-sourced to the community.

---

## 📖 Introduction

This repository provides the core PyTorch implementation of **AegisTG**, an ultralightweight RGB-D-text multimodal perception network meticulously tailored for instruction-driven object detection in embodied AGV warehousing scenarios. 

To shatter the computational barriers and severe physical degradations (such as wrapping film glare, motion blur, and dense occlusion) encountered in edge environments, AegisTG introduces three specific defense mechanisms and a novel cross-modal alignment closed loop.

## 🚀 Key Contributions (Implemented in this Repo)

This repository includes the implementation of the following core modules:

* **RAAF (Reliability-Aware Asymmetric Fusion):** Dynamically assesses spatial credibility to implement differential compensation, overcoming localized depth failures induced by glare. *(See `models/raaf.py`)*
* **OmniMixer (Omnidimensional Feature Refinement):** Applies non-linear amplitude modulation within a low-rank frequency domain to recover geometric boundaries obliterated by motion blur at an ultralow $\mathcal{O}(N \log N)$ complexity. *(See `models/omnimixer.py`)*
* **SyncUpFuse (Synchronized Upsampling Fusion):** Deploys pixel-level competitive routing to eradicate cross-scale background contamination under dense stacking. *(See `models/syncupfuse.py`)*
* **TGA (Text-Guided Alignment Network):** Cascades adaptive spatial gating and normalized dual cosine constraints to eliminate environmental amplitude interference, accomplishing a pristine manifold alignment closed loop. *(See `models/tga.py`)*

## 🛠️ Environment Requirements

* Ubuntu 20.04
* Python $\ge$ 3.8
* PyTorch $\ge$ 2.0.1
* CUDA $\ge$ 11.8

*(A complete `requirements.txt` will be provided upon full release.)*

## 🔍 Code Structure (Reviewer Guide)

For the convenience of reviewers, the core architectural implementations are organized as follows:

```text
├── models
│   ├── aegis_tg.py       # The overall architecture of AegisTG
│   ├── raaf.py           # Implementation of Reliability-Aware Asymmetric Fusion
│   ├── omnimixer.py      # Implementation of Omnidimensional Feature Refinement
│   ├── syncupfuse.py     # Implementation of Synchronized Upsampling Fusion
│   └── tga.py            # Implementation of Text-Guided Alignment Network
└── README.md             # This document
