## Introduction
Tumor-infiltrating lymphocytes (TILs), which are lymphocytes located within and around cancer cells, have been recognized as significant factors associated with the progression, prognosis, and therapy of various cancers, including breast cancer and non-small cell lung cancer. Concurrently, advancements in deep learning algorithms have substantially improved the effectiveness and precision of pathology image analysis. We conducted a search on Google Scholar using the terms ("machine learning" OR "artificial intelligence") AND "pancreatic cancer" AND "lymphocyte to tumor ratio", but no relevant literature was found. There is a pressing need for a fully automated model to evaluate the lymphocyte to tumor area ratio(LTR) in whole slide images (WSI) for clinical practice. Therefore,We developed a deep learning model for the comprehensive automated evaluation of LTR in HE-stained WSI, and validated its prognostic value in patient populations with PDAC. 


## Prerequisites
Python 3.8

environments can be `conda` create:

`conda env create -f environments.yaml`


## Training Data

For training, the data need to be arranged in the following order:
      
      Dataset
      ├── class
      │ ├── train
      │ └── test
      ...
- `class/train`: Directory containing training data for class 1.
- `class/test`: Directory containing test data for class 1.

## Train CNN

`python train.py --dataset  data_path`

## WSI_Segmentation

`python WSI_predict.py --WSI  WSI_path`


## Dataset
The original WSI images cannot be provided because these images are patient's private information.

Training data can be obtained by contacting the corresponding author.


# Example
git clone https://github.com/HaozhongMa/WSI_Quantify.git

cd WSI_Quantify

pip install -r requirements.txt
