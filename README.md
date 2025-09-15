# Lightweight Model for Automatic Audio Captioning (ICASSP 2026 Submission)

This repository provides the **official implementation** of our lightweight model for **Automatic Audio Captioning (AAC)**, submitted to **ICASSP 2026**.  
It includes training, evaluation, and caption generation code for the **Clotho** and **AudioCaps** datasets, with support for **ConvNeXt-based audio encoders** and **BART-based decoders**.  

> **Note:** Pretrained model weights will be released later to reproduce the results.


## Table of Contents
- [Installation](#installation)
- [Repository Setup](#repository-setup)  
- [Clotho Dataset](#clotho-dataset)  
- [Audiocaps Dataset](#audiocaps-dataset)  
- [Data Pre-processing](#data-pre-processing)  
- [Running Experiments](#running-experiments)  
- [Evaluation](#evaluation)  
- [AudioSet Keyword Embeddings](#-keyword-information)
- [Pretrained BART Decoder](#-pretrained-decoder)

---

## Installation

### 1. Python Environment
Make sure you are using **Python 3.10**.  
You can check your version with:

```bash
python3 --version
```

### 2. Create Virtual Environment
Create and activate a virtual environment:

**Linux/macOS:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv myenv
myenv\Scripts\activate
```

### 3. Install Requirements
Upgrade pip and install dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Setup

1. Clone the repository:

```bash
git clone  
cd AutomaticAudioCaptioning
```

2. Load the ConvNeXt checkpoint for the audio encoder (trained on the AudioSet2M). Download `convnext_tiny_471mAP.pth` from [Zenodo](https://zenodo.org/records/8020843) and place it in the `convnext` folder.  


3. Caption evaluation requires the [caption-evaluation-tools](https://github.com/audio-captioning/caption-evaluation-tools):  
-  Download and run the commands
-  Permission required use this: `chmod +x get_stanford_models.sh`
-  cocacaptions provides 
        - bleu: Bleu evalutation codes
        - meteor: Meteor evaluation codes
        - rouge: Rouge-L evaluation codes
        - cider: CIDEr evaluation codes
        - spice: SPICE evaluation codes
- Fense We provided the folder: `fense`

```bash
cd caption-evaluation-tools/coco_caption
./get_stanford_models.sh
mv ../coco_caption  ../../ #Here Make sure code files and coco_caption in the same directory
```

> **Note:** Java must be installed for the evaluation tools.

---

## Clotho Dataset

### Obtaining the Data

- Download Clotho v2.1 dataset from Zenodo: [Development, Validation, Evaluation](https://doi.org/10.5281/zenodo.4783391)  
- Download the test set (without captions) separately: [Test Set](https://zenodo.org/zenodo.3865658)  
- We already provided the csv files kindly check in `data/`
After extraction, the folder structure should look like:

```
data/
└── clotho_v2/
    ├── development/
    │   └── *.wav
    ├── validation/
    │   └── *.wav
    ├── evaluation/
    │   └── *.wav
    ├── test/
    │   └── *.wav
    ├── development_captions.csv.csv
    ├── validation_captions.csv
    └── evaluation_captions.csv
```

---

## Audiocaps Dataset

### Downloading and Structure

- Download Audiocaps using the Python package [`audiocaps-download`](https://pypi.org/project/audiocaps-download/).  
- Folder structure:

```
audiocaps/
├── train/
│   └── *.wav
├── val/
│   └── *.wav
├── test/
│   └── *.wav
├── train.csv
├── val.csv
└── test.csv
```

---

## Data Pre-processing

- Clotho pre-processing scripts:  

```bash
python clotho_dataset.py
python audio_logmel_clotho.py
```
> The script will generate in data_32k_224_mels  `<split>_audio_logmels.hdf5` and `<split>_text.csv` files ready for training and evaluation.
- Audiocaps pre-processing script:

```bash
python audio_logmel_audiocaps.py
```

> The script will generate in Audiocaps  `<split>_audio_logmels_audiocaps.hdf5` and `<split>_text.csv` files ready for training and evaluation.

---

## Running Experiments

- Experiment settings are in the `exp_settings` directory (`dcb.yaml`).  
- Run an experiment with:

```bash
python main.py --exp <exp_name> --dataset_name <dataset_name> --dataset_path <path_to_data>    
```
- `dataset_name :Clotho` `path_to_data: data_32k_224_mels`
- After training, model weights are saved in `outputs/<exp_name>_out/`.

---

## Evaluation with Pre-trained Weights

1. Download pre-trained weights from [drive](https://doi.org/10.5281/zenodo.7688773).  
2. Place the weights in the respective folders `outputs/<exp_name>_out/`.   `exp_name : Clotho`  and  `exp_name : Audiocaps` 
3. Set the workflow in `dcb.yaml`:

```yaml
workflow:
  train: false
  validate: false
  evaluate: true
```

4. Run evaluation:

- Clotho  `dataset_name: Clotho` and  `dataset_path: data_32k_224_mels`

```bash
python main.py --exp <exp_name> --dataset_name <dataset_name> --dataset_path <path_to_data>
```
- Audiocaps `dataset_name: Audiocaps` and  `dataset_path: Audiocaps`

```bash
python main.py --exp <exp_name> --dataset_name <dataset_name> --dataset_path <path_to_data>
``` 

## AudioSet Keyword Embeddings

- Every class keyword from AudioSet was tokenized into subwords, and the corresponding BART embeddings were generated and stored in: `audioset_classes_embeddings/classes_embeddings.pkl`


- To see how these embeddings were prepared, refer to the notebook: `/notebooks/custom_tokenizer.ipynb`


## Passing Keyword Information to the Model

In `models.py`, a keyword branch was added to the `forward` and `generate beam` function. This branch allows the model to incorporate AudioSet keyword embeddings during caption generation. Beam search can be applied to this setup for generating captions. 

## Using a Pretrained BART Decoder

By default, the system uses a custom 6-layer BART decoder.  
If you want to use the official pretrained `facebook/bart-base` model as the decoder, update your `dcb.yaml` configuration:

```yaml
tokenizer: facebook/bart-base
pretrained: facebook/bart-base
```
- If pretrained is set to null, the system will use the custom 6-layer BART decoder.

- If you set it to facebook/bart-base, it will load the pretrained BART weights and tokenizer for the decoder.
