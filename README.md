# Detection of Unknown Attacks Through Encrypted Traffic: A Gaussian Prototype-Aided Variational Autoencoder Framework
This branch contains audited replication fixes relative to `2554fba`.
Read [REPLICATION_NOTES.md](REPLICATION_NOTES.md) before starting training or
comparing old/new results. New training uses decoder v2; higher accuracy is not
guaranteed, and dataset/protocol equivalence to the paper remains unresolved.

IEEE TIFS 2025 ([doi:10.1109/TIFS.2025.3612141](https://doi.org/10.1109/TIFS.2025.3612141))

---

### Abstract

The identification of encrypted network traffic presents a pivotal challenge in detecting unknown malicious traffic. Unlike closed-set identification, which primarily classifies known traffic classes, detecting unknown malicious traffic necessitates both accurate classification of known traffic and the identification of previously unseen traffic classes. Existing methods often face difficulties in effectively constraining the distribution size of known classes in the representation space and frequently misclassifying unknown classes as known. To address these challenges, we propose Open-Detect, a robust theoretical framework for detecting unknown malicious traffic, which leverages advanced deep learning techniques, such as variational autoencoders and Gaussian prototypes. Open-Detect introduces two primary constraints: a generative constraint, which enhances intra-class compactness, and a discriminative constraint, which optimizes inter-class separation. These constraints collectively mitigate the risks of misclassifying known classes and failing to detect unknown classes. In Open-Detect, network flows are transformed into grayscale images, and each known traffic class is mapped to a unique Gaussian prototype in the latent space. This design ensures tight clustering of samples within the same class and clear separation of samples between different classes. The detection of unknown malicious traffic is performed based on the distance between samples and these prototypes. Extensive experiments conducted on multiple publicly available datasets substantiate the efficacy of Open-Detect. The results reveal significant improvements in intra-class compactness and inter-class separation, enabling superior performance in both closed-world and open-world scenarios, particularly for detecting unknown malicious traffic. 

![Framework Overview](README.assets/image-20250522103530795.png)
*Overview of the Open-Detect framework for unknown network traffic detection.*

---

## Table of Contents

- [Features](#features)
- [Paper-aligned implementation](#paper-aligned-implementation)
- [Dataset](#dataset)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Model Details](#model-details)
- [Results & Scenarios](#results--scenarios)

---

## Features

- **Unknown Attack Detection:** Detects both known and unknown attacks using latent Gaussian prototypes.
- **Ready-to-Run Scripts:** Includes training and evaluation scripts.

---

## Paper-aligned implementation

The training and evaluation flow follows the paper in the following places:

- Preprocessing keeps the first 8 packets per flow, with 80 header bytes and 48 payload bytes per packet, producing one 32x32 grayscale image.
- The generative constraint is reconstruction loss plus KL divergence to the correct class prototype (Equations 13 and 15).
- The discriminative constraint uses the KL divergence to every known-class prototype (Equations 17 and 18).
- The total loss is `lambda * generative + (1 - lambda) * discriminative`; there is no additional entropy term (Equation 20).
- Each repeated experiment uses a stratified 80% training, 10% validation, and 10% test split.
- The unknown-detection threshold is selected using only known validation samples so at least 95% are accepted as known (Equations 21 and 22). Unknown test samples are not used to tune it.
- `run_5fold.py` runs five seeded repetitions and reports the mean and sample standard deviation.

---

## Dataset

The dataset used for experiments is located in `data/dataset`.  
It contains network traffic from **8 different scenarios**, simulating a range of attack and normal behaviors.

You can download the dataset from Baidu Cloud:

```
Open-Detect dataset:
Link: https://pan.baidu.com/s/1DYSDeyLgDhMVHO2BAsR0aQ?pwd=8b8z 
Extraction code: 8b8z
```

**Scenarios included:**

<img src="README.assets/image-20250522105230233.png" alt="Scenarios" style="zoom: 67%;" />

Each scenario contains labeled traffic data for both benign and attack samples. The dataset is organized for easy integration with provided scripts.

---

## Quickstart

### 1. Clone the repository

```bash
git clone --branch codex/replication-fixes https://github.com/HaikalE/Open-Detect.git
cd Open-Detect
```

### 2. Download the dataset

Download and extract the dataset as described above. Place the files in `data/dataset`.

### 3. Install requirements

See [Setup & Installation](#setup--installation) for details.

---

## Usage

### Training

To train the Open-Detect model on your dataset:

```bash
python train.py --dset mal --split 0 --fold 0
```

### Testing / Evaluation

To evaluate the model (including detection of unknown attacks):

```bash
python test.py --dset mal --split 0 --fold 0
```

### Five-run evaluation

To train and evaluate five seeded 80:10:10 splits and save per-run plus summary JSON files:

```bash
python run_5fold.py --dset mal --split 0
```

These are five repeated holdouts, not classical five-fold cross-validation.
Outputs default to `save_model_v2/` and `results_v2/`. Check dataset counts first
with `python provenance.py --dset mal`. Existing checkpoint paths are never
overwritten by a new training invocation. Existing pinned notebooks remain on
their old version until their launchers and resume adapters are regenerated.

Use `--gpu -1` for CPU. Use `python train.py --help`, `python test.py --help`, or
`python run_5fold.py --help` to see all options.

---

## Project Structure

```
Open-Detect/
│
├── data/
│   └── dataset/            # Downloaded network traffic data
│   └── Preprocessing/      # Transform raw pcap file to grayscale images
├── save_model/             # Trained checkpoints
├── results/                # Per-run and mean +/- standard deviation metrics
├── model.py                # Open-Detect model and paper-aligned loss
├── train.py                # Training script
├── test.py                 # Validation-threshold and test evaluation
├── run_5fold.py            # Five seeded repetitions
├── tests/                  # Paper-alignment regression tests
├── utils.py                # Utilities
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## Setup & Installation

- **Python:** 3.10.13
- **PyTorch:** 2.1.1
- **NumPy:** 1.26.1
- **Pandas:** 2.1.3

Install dependencies (use a virtual environment for best results):

```bash
pip install -r requirements.txt
```

---

## Model Details

The core model is a **Gaussian Prototype-Aided Variational Autoencoder (Open-Detect)**.  
Key characteristics:

- **Encoder/Decoder:** Learns compact representations of network traffic.
- **Gaussian Prototypes:** Each known class is represented by one latent Gaussian prototype. Unknown classes are intentionally not assigned prototypes during training.
- **Novelty Detection:** A sample is flagged as unknown when its minimum KL distance to all known prototypes is greater than or equal to the validation threshold.

For more technical details, see the code in `model.py`.

---

## Results & Scenarios

The framework is evaluated across 8 scenarios, including multiple attack types.
The test script reports closed-world classification metrics and open-world AUROC,
accuracy, precision, recall, and F1. The five-run script reports mean +/- standard
deviation for every metric.

---

## Citation

```
@article{meng2025detection,
  title={Detection of Unknown Attacks Through Encrypted Traffic: A Gaussian Prototype-Aided Variational Autoencoder Framework},
  author={Meng, Qianwei and Tao, Jing and Yuan, Qingjun and Li, Guangsong and Wang, Yongjuan and Gao, Bing and Lu, Siqi},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  publisher={IEEE}
}
```

---

**For any questions, please open an issue or contact the authors.**
