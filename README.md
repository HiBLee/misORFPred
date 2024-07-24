# misORFPred: A Novel Method to Mine the Translatable sORFs in Plant Pri-miRNAs Using Enhanced Scalable k-mer and Dynamic Ensemble Voting Strategy
## Dataset
#### training dataset: 
* dataset/ath_training_dataset_800.txt
#### independent testing dataset: 
* dataset/ath_independent_testing_dataset1.txt
* dataset/gma_independent_testing_dataset2.txt
* dataset/vvi_independent_testing_dataset3.txt
* dataset/verified_independent_testing_dataset4.txt
## Requirement
* Python == 3.7
* biopython == 1.81
* numpy == 1.21.6
* scikit-learn == 1.0.2
* pandas == 1.3.5
## Usage
* EnsembleClassifier.py: model training and predict
* FeatureDescriptor.py: feature extraction
* FeatureSelection.py: feature selection
* MLModel.py: construct machine learning classifiers
## Citation
Li HB, Meng J, Wang ZW, Luan YS. misORFPred: A Novel Method to Mine the Translatable sORFs in Plant Pri-miRNAs Using Enhanced Scalable k-mer and Dynamic Ensemble Voting Strategy. Interdiscip Sci Comput Life Sci (2024).
