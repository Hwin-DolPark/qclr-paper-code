# QCLR: A Quantile-Based Contrastive Learning Framework for Sepsis Mortality Prediction and Clinical Decision Support Insights

**Authors:** Hwin Dol Park (park.hwin@etri.re.kr, park.hwin@kaist.ac.kr), Jae-Hun Choi (jhchoi@etri.re.kr), [Uichin Lee (uclee@kaist.ac.kr)](https://scholar.google.co.kr/citations?user=Sc2pBzYAAAAJ&hl=ko)

This repository contains the official implementation for QCLR the paper:
> Hwin Dol Park, Jae-Hun Choi, Uichin Lee. "QCLR: A Quantile-Based Contrastive Learning Framework for Sepsis Mortality Prediction and Clinical Decision Support Insights." (Under Revision, IEEE Journal of Biomedical and Health Informatics).

---

## Overview

Sepsis, a life-threatening organ dysfunction, requires accurate clinical decision support. We propose **Quantile-based Contrastive Learning Representation (QCLR)**, a novel framework for improving sepsis mortality prediction and clinical decision-making. QCLR leverages contrastive learning to generate detailed representations of patient data from electronic medical records (EMR), capturing important variations within patient cohorts. By integrating a novel **quantile-based contrastive loss** with a mortality prediction loss, QCLR jointly optimizes representation learning and predictive accuracy.

**Key Contributions:**
* **Novel QCLR Loss**: Introduces a quantile-based contrastive loss function that captures intra-class patient variations by dynamically defining positive and negative pairs using similarity quantiles calculated exclusively within specific outcome groups.
* **Joint Optimization**: Integrates this specialized contrastive learning with the clinical objective of mortality prediction via a joint optimization strategy, ensuring representations are both discriminative for similarity assessment and optimized for prognostic features.
* **Actionable Clinical Decision Support**: Leverages these learned representations to enable the effective retrieval of clinically similar historical patient cases, offering insights for treatment decisions.

---

## Framework Overview

The workflow of the QCLR framework is illustrated below:

<p align="center">
  <img src="fig/fig1.pdf" alt="QCLR Workflow" width="800"/>
  </p>
<p align="center">
  <em><b>Figure 1:</b> Workflow of the QCLR framework. Input patient data (X) is processed by the Sequence Augmentation Module to generate X', and both are fed to the Medical Encoder, yielding a core representation <b>z</b>. This representation <b>z</b> serves dual pathways: (i) the Mortality Decoder uses it for outcome prediction (ŷ<sub>i</sub>) via a binary cross-entropy loss (𝓛<sub>BCE</sub>), and (ii) the Similarity Projector maps <b>z</b> to an embedding <b>v</b> where our novel QCLR loss (L<sub>QCLR</sub>) is applied. The L<sub>QCLR</sub> refines these embeddings by contrasting positive and negative pairs selected based on similarity quantiles within the same outcome class. Through joint optimization of 𝓛<sub>BCE</sub> and L<sub>QCLR</sub>, the trained model can predict mortality risk for new patients and retrieve clinically similar historical cases.</em>
</p>

---

## Datasets

QCLR was evaluated on three distinct datasets:

### 1. ASAN Sepsis Dataset
* **Description**: A private dataset comprising 30,874 sepsis cases (14,843 unique patients after preprocessing) retrospectively collected from adult patients at Asan Medical Center, Seoul, South Korea (2009-2020). It includes demographics, vital signs, lab results, and treatments (71 features). The outcome is 28-day mortality.
* **Access**: Due to IRB restrictions (Seoul Asan Medical Center IRB number: 2021-15-050) and patient privacy, the ASAN dataset cannot be publicly shared.
* **Preprocessing**: Missing values were imputed (forward/backward fill), numerical features min-max normalized ([0,1]), and sequences zero-padded. A structured random subsequence selection strategy was used for augmentation and class imbalance handling (survivors augmented 3x, deceased 6x). Details can be found in Section IV-A, IV-B, and Appendix B of our paper.

### 2. MIMIC-III Sepsis Cohort
* **Description**: A publicly available dataset of ICU patient records. We constructed a sepsis cohort based on Sepsis-3 criteria, resulting in 12,252 sepsis patients. It includes 41 features (demographics, clinical status, treatments). The outcome is 90-day mortality.
* **Access**: Available on PhysioNet: [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/).
* **Preprocessing**: Similar to the ASAN dataset: imputation, min-max normalization, zero-padding. Augmentation was applied (survivors 3x, deceased 5x). Data was extracted and processed following protocols similar to those described in Komorowski et al., 2018. Details in Section IV-A and IV-B of our paper.

### 3. PTB Diagnostic ECG Database
* **Description**: Used for myocardial infarction (MI) classification against healthy controls. ECG recordings from 198 subjects (15 channels) were downsampled, normalized, and segmented into single heartbeats, yielding 64,356 samples.
* **Access**: Available on PhysioNet: [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/).
* **Preprocessing**: Standardized preprocessing protocols analogous to those in `Medformer` were used, including subject-independent splits (60/20/20% for train/val/test). Details in Section IV-D of our paper.
* The processed PTB dataset can be manually downloaded on the github repository: https://github.com/DL4mHealth/Medformer

---

## Experimental Setups

We compare with 5 time series transformer baselines: Transformer, PatchTST, MTST, iTransformer, MEDFORMER. For a relatively fair comparison, we implemented our QCLR method and all baselines based on the MEDFORMER project at the University of North Carolina-Charlotte.

---

## Requirements

The main requirements are:
* Python 3.9
* einops 0.8.1
* matplotlib 3.7.0
* numpy 1.23.5
* pandas 1.5.3
* patool 1.12
* reformer-pytorch 1.4.4
* scikit-learn 1.2.2
* scipy 1.10.1
* sktime 0.16.1
* sympy 1.13.0
* torch 2.6.0+cu118
* tqdm 4.64.1
* wfdb 4.3.0
* neurokit2 0.2.10

To install all dependencies, first clone the repository and then use the provided `requirements.txt` file:
```bash
git clone [https://github.com/Hwin-DolPark/qclr-paper-code.git](https://github.com/Hwin-DolPark/qclr-paper-code.git)
cd qclr-paper-code
pip install -r requirements.txt
```

---

## Run Experiment
All experiments can be run using the run.py script in the run_example folder with appropriate configuration.

---

## Acknowledgement
The codebase for some of the benchmark models and experimental framework was adapted from or inspired by the following repositories:

DL4mHealth/Medformer
DL4mHealth/COMET
We thank the authors for making their code publicly available.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

