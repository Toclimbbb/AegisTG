### 🛡️ AegisTG: Anti-Degradation Detection Framework Driven by Textual Instructions for Embodied AGVs

**Official PyTorch Implementation** (Accepted by ICIG 2026)

---

#### 📖 Introduction

This repository provides the core PyTorch implementation of AegisTG, an ultralightweight RGB-D-text multimodal perception network meticulously tailored for instruction-driven object detection in embodied AGV warehousing scenarios. 

To achieve real-time deployment on edge-constrained platforms and overcome severe physical degradations (such as wrapping film glare, motion blur, and dense occlusion) encountered in edge environments, AegisTG introduces three specific defense mechanisms and a novel cross-modal alignment closed loop. 

#### 🚀 Key Contributions

This repository includes the implementation of the following core modules:
*   **RAAF (Reliability-Aware Asymmetric Fusion):** Dynamically assesses spatial credibility to implement differential compensation, overcoming localized depth failures induced by glare. (See `models/raaf.py`)
*   **OmniMixer (Omnidimensional Feature Refinement):** Applies non-linear amplitude modulation within a low-rank frequency domain to recover geometric boundaries obliterated by motion blur at an ultralow $\mathcal{O}(N\log N)$ complexity. (See `models/omnimixer.py`)
*   **SyncUpFuse (Synchronized Upsampling Fusion):** Deploys pixel-level competitive routing to eradicate cross-scale background contamination under dense stacking. (See `models/syncupfuse.py`)
*   **TGA (Text-Guided Alignment Network):** Cascades adaptive spatial gating and normalized dual cosine constraints to eliminate environmental amplitude interference, accomplishing a pristine manifold alignment closed loop. (See `models/tga.py`)

#### 🛠️ Environment Requirements

*   Ubuntu 20.04
*   Python $\geq$ 3.8
*   PyTorch $\geq$ 2.0.1
*   CUDA $\geq$ 11.8

*(Note: Please install dependencies using `pip install -r requirements.txt`)*

#### 🔍 Code Structure

The core architectural implementations are organized as follows:
*   `models/aegis_tg.py`: The overall architecture of AegisTG
*   `models/raaf.py`: Implementation of Reliability-Aware Asymmetric Fusion
*   `models/omnimixer.py`: Implementation of Omnidimensional Feature Refinement
*   `models/syncupfuse.py`: Implementation of Synchronized Upsampling Fusion
*   `models/tga.py`: Implementation of Text-Guided Alignment Network

#### 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{li2026aegistg,
  title={Anti-Degradation Detection Framework Driven by Textual Instructions for Embodied AGVs},
  author={Li, Pan and Huang, Xixia and Li, Haobin},
  booktitle={International Conference on Image and Graphics (ICIG)},
  year={2026}
}
