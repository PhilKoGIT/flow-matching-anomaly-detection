# Unsupervised Anomaly Detection with Flow-Matching Models on Tabular Data

This is the implementation of the thesis with the title "Unsupervised Anomaly Detection with Flow-Matching Models on Tabular Data".

Different scoring functions were implemented in:
- [TCCM](https://github.com/ZhongLIFR/TCCM-NIPS/tree/main)
- [ForestFlow](https://github.com/SamsungSAILMontreal/ForestDiffusion)
- ForestDiffusion (same framework as ForestFlow)

---

## Contents

This repo includes additionally to the code of the thesis:
- The prompts for the LLM application in `prompts_llm_application.json`
- The plots of the studies (Main, Sensitivity studies of all models, business dataset study)
- The generator for generating the `business_transaction_dataset`

---

## Datasets

- Campaign and business dataset can be found on the USB stick
- Additionally:
  - **Campaign dataset:** Public or available [here](https://github.com/ZhongLIFR/TCCM-NIPS/tree/main/datasets/high_dim)
  - **Business dataset:** Can be generated (but due to random effects it won't be exactly the same as in the thesis). After generation, move it into `data_contamination`

---

## Installation

### Requirements

- Python 3.11

### Setup

**1. Clone repository**

```bash
git clone https://github.com/PhilKoGIT/flow-matching-anomaly-detection.git
cd flow-matching-anomaly-detection
```

**2. Create virtual environment**

```bash
python3.11 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip setuptools wheel
```

**3. Install dependencies**

```bash
pip install -r requirements_mac.txt
```

**4. Install ForestDiffusion**

```bash
cd Python-Package/base-ForestDiffusion
pip install -e .
cd ../..
```

**5. Test installation**

```bash
python -c "import torch; import ForestDiffusion; print('OK!')"
```

---

## Usage

### Training the Models

> For configuration, please look into the description in the file.

```bash
cd scripts
python contamination.py
```

**OR** run with cluster:

```bash
sbatch run_contamination_cpu.sbatch
```

### Generate Plots

> For configuration, please look into the description in the file.

```bash
python plot_and_merge.py
```

### Generate Plots

> For configuration, please look into the description in the file.

```bash
python plot_and_merge.py
```

### Generate Business Dataset

```bash
cd generator
python generator.py
```

> After generation, move the dataset into `data_contamination`.