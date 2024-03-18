# Deciphering Mutational Impacts on Predicting Drug-Target Affinity through Interpretable Transfer Learning

MutDTA is an interpretable transfer learning model designed for predicting mutational effects on drug-target affinity. MutDTA employs D-MPNN and 1D-CNN to extract embeddings from drug SMILES and target sequences, respectively. Central to our model is the integration of a cross-attention mechanism, skillfully fusing drug and target embeddings. To address the challenge of predicting mutational effects on DTA, we implement transfer learning with MutDTA. The model undergoes an initial pre-training phase on a benchmark dataset, assimilating foundational DTA-related knowledge. Subsequently, a fine-tuning phase on the platinum dataset, a meticulously curated collection of high-quality mutation data, allows the model to acquire mutation-specific insights. In conducting a detailed analysis of the results from the platinum test set, we found that MutDTA effectively identifies how mutations in targets related to pathogens affect DTA. By using real-world datasets of patients with moderate to severe COVID-19, MutDTA successfully identified mutation sites that induce drug resistance to approved anti-SARS-CoV-2 drugs such as nirmatrelvir. This highlights the practical value of MutDTA in addressing challenges in drug resistance.

# Requirements
```
torch 1.8.0
python 3.8.18
numpy 1.22.0
pandas 1.3.1
scikit-learn 0.24.0
scipy 1.10.1
```

# Dataset
We systematically collected and integrated the benchmark dataset from three common datasets for drug-target binding prediction: DAVIS, BindingDB, and BioLip. BindingDB and BioLip dataset each comprise a set of drug target pairs, accompanied by their binding affinity scores, which are quantified using one of the following metrics: inhibition constant ($K_{i}$), dissociation constant ($K_{d}$), or inhibitory concentration 50 (\(IC_{50}\)). The DAVIS dataset exclusively features drug-target pairs with affinity scores based on ($K_{i}K_{d}$) metrics. For this study, we selectively utilized ($K_{d}$) values from these datasets as benchmark dataset. We conducted thorough data cleansing and deduplication, resulting in a comprehensive dataset that includes 87,508 ($K_{d}$) values, encompassing interactions between 21,654 drugs and 2,690 targets.

The platinum dataset is a manually collected, literature-driven, high-quality mutation dataset, encompassing over 1,000 mutations. It uniquely correlates experimental data on affinity alterations with the three-dimensional structures of protein-ligand complexes. We searched from relevant literature, downloaded the relevant pdb sequences from the RCSB Protein Data Bank website(https://www.rcsb.org/), and manually collected relevant mutation data from literatures. After processing this data, we obtained a total of 1189 pairs, involving 644 targets and 173 drugs, each paired with corresponding affinity values. In line with the methodology outlined by, we converted the affinity values to a log space as follows:
$$pK_{d} = -\log_{10}\left(\frac{K_{d}}{1 \times 10^9}\right)$$

# Run the demo
```
python main.py
```
