# MutDTA: Unveiling Resistance Mechanisms of Viral Protein Mutations via Interpretable Transfer Learning

This repository contains the MutDTA deep learning model, which is a transfer learning model designed to predict the impact of mutations on DTA and to elucidate resistance mechanisms effectively. The model is implemented in python and pytorch for accurate affinity predictions.

# Requirements
```
torch 1.8.0
python 3.8.18
numpy 1.22.0
pandas 1.3.1
scikit-learn 0.24.0
scipy 1.10.1
```

# Datasets
All publicly datasets used can be accessed here:
| dataset      | url |
| ----------- | ----------- |
| DAVIS      | https://github.com/hkmztrk/DeepDTA/tree/master/data/davis       |
| BindingDB   | https://www.bindingdb.org/rwd/bind/index.jsp        |
| BioLip   | https://zhanggroup.org/BioLiP/index.cgi        |
| Plaitnum | https://biosig.lab.uq.edu.au/platinum |

Or you can use our preprocessing data:

| dataset      | url |
| ----------- | ----------- |
| pre-training      | https://github.com/altriavin/MutDTA/blob/master/dataset/Kd.csv       |
| fine-tuning   | https://github.com/altriavin/MutDTA/blob/master/dataset/platinum.csv        |

# Usage
1. Train and test on the demo data
```
python main.py --dataset <data_name> --learn_rate <learn_rate> --epochs <epochs> --batch_size <batch_size>
```
For example:
```
python main.py --dataset platinum --learn_rate 0.00001 --epochs 100 --batch_size 128
```
2. Train and test on your own data

If you want to run MutDTA on your data, just preprocess your data into the specified form as follows:

| smile      | seq | labels |
| ----------- | ----------- | ----------- |
| drug_smile  | protein_seq | -log($\frac{affinity}{1e9}$) |

and then, you can use MutDTA to train or test your data using the following command:
```
python main.py --dataset <your_dataset_name> --learn_rate 0.00001 --epochs 100 --batch_size 128
```

# Contact
If you have any questions, feel free to contact me by email: vinaltria@csu.edu.cn