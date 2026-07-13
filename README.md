# Automatic Endometriosis Lesion Segmentation in Laparoscopy

This repository provides a benchmarking framework for automatic and promptable segmentation of endometriosis lesions in laparoscopic images. The project compares supervised segmentation models, SAM-based foundation models, and hybrid auto-prompting approaches across multiple endometriosis datasets.

The goal is to evaluate how well different model families segment visually heterogeneous, small, and ambiguous endometriosis lesions, and whether promptable foundation models can improve fully automatic segmentation pipelines.

---

## Reference

This repository accompanies the following work:

**Evaluating the Prompt-to-Automation Gap: A Cross-Dataset Benchmark for Laparoscopic Endometriosis Segmentation** (in work of publication, submitted to CAPI workshop, MICCAI 2026)

Please cite the paper if you use this repository, code, or results in your own research.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Datasets](#datasets)
- [Models](#models)
- [Hybrid Segmentation](#hybrid-segmentation)
- [Evaluation](#evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Project Overview

Endometriosis lesion segmentation in laparoscopic surgery remains challenging due to large appearance variability, weak contrast, irregular lesion boundaries, specular highlights, blood, fibrosis, and visual similarity to surrounding tissue.

This project benchmarks several segmentation strategies:

- fully supervised segmentation models,
- zero-shot SAM-based promptable models,
- prompt-type comparisons using point, box, and combined prompts (box+point, box+positive+negative points)
- hybrid automatic pipelines using best performing trained model (SegFormer) predictions as prompts for SurgiSAM2 refinement.

The benchmark focuses on cross-dataset evaluation, prompt sensitivity, and the practical limitations of using foundation models for fully automatic lesion segmentation.

---

## Features

- **Supervised segmentation benchmarking** using CNN-, Transformer-, and YOLO-based models.
- **SAM-based prompt comparison** across point, box, box+point, and box+positive/negative prompts.
- **Hybrid auto-prompting pipelines** using SegFormer predictions to generate SurgiSAM2 box prompts.
- **Fallback-based refinement** to avoid harmful SAM corrections when candidate masks disagree with the supervised prediction.
- **Image-level metric calculation** for Dice, IoU, precision, and recall.
- **Statistical comparison scripts** using paired Wilcoxon signed-rank tests and Holm correction.
- **Publication-quality plots** for model comparison, prompt comparison, and qualitative visualization.
- **Cross-dataset evaluation** on ENID, GLENDA, and GLENDA-clean.

---

---

## Repository Structure

```text
Auto_Endometriosis_Segmentation/
│
├── configs/
│   └── Configuration files and path/model settings
│
├── Hybrid/
│   └── Hybrid SegFormer + SAM/SurgiSAM2 refinement utilities
│
├── Model_comparison/
│   └── Scripts for comparing models, prompts, statistics, inference time, and figures
│
├── Preprocess/
│   └── Dataset preprocessing and standardization scripts
│
├── run_models/
│   └── Main executable scripts for running training/inference pipelines
│
├── scripts/
│   └── Additional utility scripts
│
├── src/
│   └── Core source code, model utilities, datasets, metrics, and helper functions
│
├── LICENSE
│
└── README.md
```

---

## Datasets

The benchmark uses laparoscopic endometriosis segmentation datasets:

- **ENID**
- **GLENDA**
- **GLENDA-clean**

GLENDA-clean excludes box-like annotations and artifacts from the original GLENDA dataset to reduce annotation-artifact bias during training and evaluation.

Each dataset is organized into standardized train, validation, and test splits with patient-level separation where applicable.

---

## Models

### Supervised Models

The following supervised segmentation models are included:

- DeepLabV3+
- UNet++
- nnU-Net v2
- SegFormer
- YOLO11-seg

### SAM-Based Models

The following promptable foundation models are evaluated without task-specific fine-tuning:

- SAM2
- SurgiSAM2
- MedSAM
- SAM-Med2D

Prompt modes include:

- point prompt,
- box prompt,
- box + positive point,
- box + positive and negative points.

---

## Hybrid Segmentation

The repository includes hybrid automatic segmentation pipelines that combine supervised predictions with SAM-based refinement.

The main hybrid setup is:

1. Train or load a supervised SegFormer model.
2. Extract connected components from the SegFormer prediction.
3. Convert each component into an automatic prompt.
4. Run SurgiSAM2 using box or box+positive/negative prompts.
5. Accept the SurgiSAM2 candidate only if it agrees sufficiently with the SegFormer component.
6. Otherwise, fall back to the original SegFormer component.

Implemented hybrid variants include:

- `SegFormer_SurgiSAM2_AutoBox`
- `SegFormer_SurgiSAM2_AutoBox_Fallback`
- `SegFormer_SurgiSAM2_AutoBox_PosNeg_Fallback`

---

## Evaluation

The main metrics are:

- Dice score
- Intersection over Union (IoU)
- Precision
- Recall

Statistical comparisons are performed with paired image-level tests:

- paired Wilcoxon signed-rank test,
- Holm-Bonferroni correction,
- mean and median paired differences,
- per-image improvement and worsening rates.

The evaluation scripts generate CSV and Excel summaries for prompt comparisons, supervised model comparisons, and hybrid refinement analyses.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Auto_Endometriosis_Segmentation.git
cd Auto_Endometriosis_Segmentation
```
Create and activate a Python environment (preferrably a separate rnv for each model to avoid any conflicts):
```
python -m venv sam2-env
sam2-env\Scripts\activate
```
Install the required packages:
```
pip install -r requirements.txt
```
Additional SAM/SurgiSAM2 setup requires downloading external model checkpoints and placing them in the expected checkpoint folders.

## Usage 

Usage example:
Run supervised model inference
```
python run_models/run_segformer.py
```
Run hybrid SegFormer + SurgiSAM2 inference
```
python run_models/run_hybrid_segformer_surgisam2_autobox_fallback.py
```
Run box+positive/negative fallback hybrid inference
```
python run_models/run_hybrid_segformer_surgisam2_autobox_posneg_fallback.py
```
Run prompt comparison analysis
```
python Model_comparison/prompt_comparison.py
```
Run hybrid statistical comparison
```
python Model_comparison/compare_posneg_hybrid_statistics.py
```

The generated outputs include image-level metrics, summary Excel files, visual comparison plots, and qualitative segmentation examples.

## License

This project is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at:

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

