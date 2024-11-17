# MutDTA: Unveiling Resistance Mechanisms of Viral Protein Mutations via Interpretable Transfer Learning

MutDTA is an interpretable transfer learning model designed to predict mutational effects on drug-target affinity. MutDTA utilizes D-MPNN and CNN to extract embeddings from drugs SMILE and targets sequence, respectively. Central to our model is the integration of a cross-attention mechanism, which adeptly fuses the embeddings of drug and target. Addressing the challenge of predicting mutational effects on DTA, we introduce the transfer learning with MutDTA. The model undergoes an initial phase of pre-training on a pretrain dataset, enabling it to assimilate foundational DTA-related knowledge. This is followed by a fine-tuning phase on the platinum dataset, a meticulously curated collection of high-quality mutation data, facilitating the model's acquisition of mutation-specific insights. A key feature of MutDTA is its use of the cross-attention mechanism to generate interaction score matrices, which elucidate the interactions between drug atoms and amino acids in the target sequence. This capability allows our model to precisely identify the binding sites of drug-target interactions, significantly enhancing the interpretability of its predictions.

# Requirements
```
torch 1.8.0
python 3.8.18
numpy 1.22.0
pandas 1.3.1
scikit-learn 0.24.0
scipy 1.10.1
```

# Run the demo
```
python main.py
```
