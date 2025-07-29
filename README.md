# RAGP: A Retrieval-Augmented Deep Learning Model for Genomic Prediction in Crop Breeding

## 1. Introduction

This is the code for the paper: **"RAGP: A retrieval-augmented deep learning model for genomic prediction in crop breeding"**. RAGP introduces a retrieval-augmented mechanism to enhance genomic prediction by incorporating references from genetically similar individuals. This method significantly improves performance, especially under small sample sizes and complex population structures.
![image](xxx.png)
---

## 2. Dataset

The following datasets are supported:
- **wheat599**
- **wheat2000 and maize8652**: Download from Baidu Cloud:  
  🔗 https://pan.baidu.com/s/1qorIcAyx6tOJSBSjMP8hLA  
  🔑 Extraction Code: `0720`


Please place the datasets in the appropriate folders (e.g., `./data/`) as expected by the configuration files.

---

## 3. Requirements

Install the following Python packages with specified versions:

```bash
torch==1.13.1
torchvision==0.14.1
numpy==1.26.0
tqdm
scipy
scikit-image
pandas
```

Recommended Python version: `>=3.7`

---

## 4. Running the Model

All configuration files are located in the `RAGP/config/` folder.

To run the model on the `wheat599` dataset, use the following command:

```bash
python RAGP/run.py --config ./config/config_wheat599.json
```

The model will train and evaluate, and the resulting model weights for each task will be saved in: `RAGP/ckpt/`

---

## 5. Generating Retrieval References

To generate the reference individuals used during testing, run:

```bash
python RAGP/references.py
```

The generated references will be saved in:
`RAGP/references/`

---

## Citation

If you use this work in your research, please cite our paper


---


For questions or issues, feel free to open an issue or contact the authors.
