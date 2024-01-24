# Exploring Mutational Effects on Drug-Target Affinity Prediction using An Interpretable Transfer Learning

MutDTA is an interpretable transfer learning model designed to predict mutational effects on drug-target affinity. MutDTA utilizes D-MPNN and CNN to extract embeddings from drugs SMILE and targets sequence, respectively. Central to our model is the integration of a cross-attention mechanism, which adeptly fuses the embeddings of drug and target. Addressing the challenge of predicting mutational effects on DTA, we introduce the transfer learning with MutDTA. The model undergoes an initial phase of pre-training on a benchmark dataset, enabling it to assimilate foundational DTA-related knowledge. This is followed by a fine-tuning phase on the platinum dataset, a meticulously curated collection of high-quality mutation data, facilitating the model's acquisition of mutation-specific insights. Subsequently, we apply MutDTA to the SARS-CoV-2 datasets, evaluating its efficacy in practical scenarios. A key feature of MutDTA is its use of the cross-attention mechanism to generate interaction score matrices, which elucidate the interactions between drug atoms and amino acids in the target sequence. This capability allows our model to precisely identify the binding sites of drug-target interactions, significantly enhancing the interpretability of its predictions.

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
$$
    pK_{d} = -\log_{10}\left(\frac{K_{d}}{1 \times 10^9}\right)
$$

The SARS-CoV-2 dataset (in the case folder) utilized in our study is sourced from the research documented in. This reference provides a detailed determination of the inhibition constant ($K_{i}$) values for various antiviral agents, including GC-376, PF-00835231, and Nirmatrelvir (PF-07321332), against both the wild-type and mutant proteins of SARS-CoV-2, assessed via FRET assay. Our dataset comprises 120 pairs of affinity values, encompassing interactions between 52 distinct SARS-CoV-2 and related mutant sequences and three specific drugs. Moreover, to facilitate a more nuanced analysis, these affinity values were transformed into logarithmic space, adhering to the equation 1.

# Run the demo
```
python main.py
```