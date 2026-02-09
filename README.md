<p align="center">
  <img src="assets/MuViS_Banner.png" alt="MuViS Pipeline" width="1000"/>
</p>

# MuViS: Multimodal Virtual Sensing Benchmark

This repository contains the **MuViS** codebase - dataset preprocessing, unified time-series I/O, configuration driven experiment runners and logging utilities for reproducible benchmarking across datasets. The corresponding paper is available at: \<PLACEHOLDER>

> Abstract: Virtual sensing infers hard-to-measure quantities from accessible measurements and is central to perception and control in physical systems. Despite rapid progress from first-principle and hybrid models to modern data-driven methods research remains siloed, leaving no established default approach that transfers across processes, modalities, and sensing configurations. We introduce \textsc{MuViS}, a domain-agnostic benchmarking suite for multimodal virtual sensing that consolidates diverse datasets into a unified interface for standardized preprocessing and evaluation. Using this framework, we benchmark representative approaches spanning gradient-boosted decision trees and deep neural network (NN) architectures, and quantify how close current methods come to a broadly useful default. \textsc{MuViS} is released as an open-source, extensible platform for reproducible comparison and future integration of new datasets and model classes.

## Overview

Virtual sensing aims to infer hard-to-measure quantities from accessible primary measurements and is central to perceiving and controlling physical systems. Despite rapid progress, research is typically siloed in narrow application domains, limiting insight into how well approaches generalize.

**MuViS** is a comprehensive, domain-agnostic benchmarking suite for multimodal virtual sensing. It addresses the heterogeneity in file formats, split definitions, and sequence lengths by providing a framework that:

- **Standardizes data preprocessing:** Converts raw datasets from `data/raw/<dataset>/` into a consistent [`.ts` format](https://www.sktime.net/en/stable/api_reference/file_specifications/ts.html) in `data/processed/<dataset>/` with predefined train-test splits (`train.ts`, `test.ts`), uniform sample shapes (X: N×T×C, y: N), and consistent missing value treatment.
- **Enables reproducible experiments:** Provides config-driven training pipelines to systematically benchmark neural networks and tree-based models across multiple datasets.

Model-specific preprocessing operations (e.g., standardization, sequence flattening) are performed within training scripts to maintain flexibility in model architecture design.

## Datasets

<p align="center">
  <img src="assets/Figure_2.png" alt="MuViS Pipeline" width="1000"/>
</p>

MuViS aggregates six benchmark datasets spanning environmental monitoring, health sensing, vehicle dynamics, tire thermodynamics, chemical process monitoring, and electrochemical energy systems.

| Dataset | Domain | Target | Inputs | Features ($D$) | Steps ($T$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Beijing Air Quality** | Environmental | PM2.5 / PM10 | Pollutants & Meteorology | 9 | 24 |
| **Revs Program** | Automotive | Lateral Velocity ($v_y$) | Driver inputs, IMU, Wheel speeds | 12 | 20 |
| **Tire Temperature** | Automotive | Tire Temp ($t_{tire}$) | Vehicle motion, Control inputs | 11 | 50 |
| **Tennessee Eastman** | Industrial | Chemical Conc. | Process vars & Manipulated vars | 33 | 20 |
| **Panasonic 18650PF** | Energy | State-of-Charge (SoC) | Voltage, Current, Temp | 7 | 500 |
| **PPG-DaLiA** | Health | Heart Rate (BPM) | BVP, EDA, Temp, Accel | 6 | 256 |

## Baselines

We benchmark representative learning approaches spanning gradient-boosted decision trees and deep neural network (NN) architectures:

- **Tree-based**: XGBoost, CatBoost
- **Neural Networks**: MLP, ResNet1D, LSTM, Transformer

## Prerequisites

- Python >=3.13
- Set up the environment and install MuViS.

```bash
# Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and the package
pip install -r requirements.txt
pip install -e .
```

- **Download the datasets** and place them in the `data/raw/` folder as described below:

  - [BeijingPM10Quality](https://zenodo.org/records/3902667)
  - [BeijingPM25Quality](https://zenodo.org/records/3902671)
  - [Panasonic18650PFData](https://data.mendeley.com/datasets/xf68bwh54v/1) Note: Panasonic_NCR18650PF_Data_Normalized.zip
  - [PPGDalia](https://archive.ics.uci.edu/dataset/495/ppg+dalia)
  - [REVS/2013_Monterey_Motorsports_Reunion](https://purl.stanford.edu/tt103jr6546)
  - [REVS/2013_Targa_Sixty_Six](https://purl.stanford.edu/yf219gg2055)
  - [REVS/2014_Targa_Sixty_Six](https://purl.stanford.edu/hd122pw0365)
  - [TennesseeEastmanProcess](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1)
  - [VehicleDynamicsDataset](https://exhibits.stanford.edu/data/catalog/hh613qz0317) Note: Store November and October sessions in separate folders.

- Final directory structure should look like this:

```text
MuViS/
├── ...
├── data/
│   ├── raw/                             
│   │   ├── BeijingPM10Quality/
│   │   │   ├── BeijingPM10Quality_TEST.ts
│   │   │   └── BeijingPM10Quality_TRAIN.ts
│   │   ├── BeijingPM25Quality/
│   │   │   ├── BeijingPM25Quality_TEST.ts
│   │   │   └── BeijingPM25Quality_TRAIN.ts
│   │   ├── Panasonic18650PFData/
│   │   │   ├── Normalization/
│   │   │   ├── Test/
│   │   │   ├── Train/
│   │   │   └── Validation/
│   │   ├── PPGDalia/
│   │   │   ├── PPG_FieldStudy/
│   │   │   ├── data.zip
│   │   │   └── readme.pdf
│   │   ├── REVS/
│   │   │   ├── 2013_Monterey_Motorsports_Reunion/
│   │   │   │   └── *.csv
│   │   │   ├── 2013_Targa_Sixty_Six/
│   │   │   │   └── *.csv
│   │   │   └── 2014_Targa_Sixty_Six/
│   │   │       └── *.csv
│   │   ├── TennesseeEastmanProcess/
│   │   │   ├── TEP_FaultFree_Testing.RData
│   │   │   └── TEP_FaultFree_Training.RData
│   │   ├── VehicleDynamicsDataset/
│   │   │   ├── Nov2023/
│   │   │   │   └── *.csv
│   │   │   └── Oct2023/
│   │   │       └── *.csv
│   │   └── ...
│   └── processed/                       
└── ...
```

## Preprocessing

To preprocess the raw datasets into the standardized `.ts` format, run:

```bash
python src/muvis/data_utils/preprocess.py
```

## Run the training

#### Single experiment

Execute the following command to run a single experiment:

```bash
python main.py single --runconf configs/<DATASET_NAME>/<MODEL_NAME>.yaml
```

#### Multiple Experiments

To run multiple experiments and save the results to a CSV file, use the command below:

```bash
python main.py batch \
  --configs \
    configs/<DATASET_NAME_1>/<MODEL_NAME_1>.yaml \
    configs/<DATASET_NAME_2>/<MODEL_NAME_2>.yaml \
  --metric test_rmse \
  --output experiment_results.csv
```

## Reproduce Results

Our evaluation demonstrates that while gradient-boosted ensembles remain highly competitive, the landscape is nuanced, with specific NN architectures excelling in distinct domains. No single architecture attains a statistically superior edge across the entire benchmark, underscoring the need for specialized architectures in virtual sensing.

To reproduce the results from the paper you can run:

```bash
bash run.sh
```

## Contributing

## Add your own dataset

Each dataset must ultimately produce two files:

- `train.ts`
- `test.ts`

---

### Step 1. Place Raw Data

Copy your raw dataset files into:
`data/raw/<YourDatasetName>/`
> **Note:**  MuViS does not impose any restrictions on the raw data format.

### Step 2: Add a Dataset Converter

MuViS handles different raw dataset formats by converting them into a common
`.ts` representation using dataset-specific converters.
All converters live in:

[`src/muvis/data_utils/converters.py`](src/muvis/data_utils/converters.py)

Each dataset is implemented as a subclass of `BaseConverter`. To add a new dataset, create a new class that inherits from `BaseConverter` and implement the `load_raw()` method.

At a minimum, every converter must:

1. Read raw files from `data/raw/<YourDataset>/`
2. Split data into train and test sequences
3. Generate fixed-length sliding windows
4. Return data in MuViS’s internal case format

### Step 3: Run Preprocessing

Once the converter is implemented, add it to the command-line interface at [src/muvis/data_utils/preprocess.py](src/muvis/data_utils/preprocess.py) and run:

```bash
python src/muvis/data_utils/preprocess.py --dataset <YourDatasetName> 
```

## Add your own model

MuViS supports both neural and tree-based models.

### Step 1: Implement the Model

Model implementations are located in the following files:

- **Neural networks:** Define your model architecture in [src/muvis/utils/architectures.py](src/muvis/utils/architectures.py)
- **Tree-based models:** Add your model to the model dictionary in [src/muvis/train/run_tree_experiments.py](src/muvis/train/run_tree_experiments.py). Any model following the scikit-learn `.fit()` convention is supported.

### Step 2: Create a Configuration File

Each experiment is controlled via a YAML configuration file. Create `configs/<YourDatasetName>/<YourModelName>.yaml` specifying the model type, hyperparameters, and training settings.
