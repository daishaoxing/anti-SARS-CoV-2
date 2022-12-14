# anti-SARS-CoV-2
The code and associated dataset for the study of “In silico identification of anti-SARS-CoV-2 medicinal plants using cheminformatics and machine learning”
Abstract: 
Severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative pathogen of COVID-19, is spreading rapidly and has caused hundreds of millions of infections and millions of deaths worldwide. Due to the lack of specific vaccines and effective treatments for COVID-19, there is an urgent need to identify effective drugs. Traditional Chinese medicine (TCM) is a valuable resource for identifying novel anti-SARS-CoV-2 drugs based on the important contribution of TCM and its potential benefits in  COVID-19 treatment. Herein, we aimed to discover novel an-ti-SARS-CoV-2 compounds and medicinal plants from TCM by establishing a prediction method of anti-SARS-CoV-2 activity using machine learning methods. We firstly constructed a benchmark dataset from anti-SARS-CoV-2 bioactivity data collected from the ChEMBL database. Then, we established random forest (RF) and support vector machine (SVM) models that both achieved satisfactorily predictive performance with AUC values of 0.90. By using this method, a total of 1013 active anti-SARS-CoV-2 compounds were predicted from the TCMSP database. Among these compounds, six compounds with highly potent activity were confirmed in the anti-SARS-CoV-2 experiments. The molecular fingerprint similarity analysis revealed that only 24 of the 1013 compounds have high similarity to the FDA-approved antiviral drugs, indicating that most of the compounds were structurally novel. Based on the predicted anti-SARS-CoV-2 compounds, we identified 74 anti-SARS-CoV-2 medicinal plants through enrichment analysis. These identified plants are widely distributed in 68 genera and 43 families. In summary, this study provided several medicinal plants with potential anti-SARS-CoV-2 activity, which offer an attractive starting point and a broader scope to mine for potentially novel anti-SARS-CoV-2 drugs.

Introduction

This repository contains an anti-SARS-CoV-2 compound predictor constructed based on machine learning methods for screening novel anti-SARS-CoV-2 drugs from large compound database.


Requirement
the predictor is a program developing with Python. Pybel, a python wrapper of Openbabel, was used to deal with compounds and generate molecular fingerprints for compounds. scikit-learn package was used to train and generate anti-SARS-CoV-2 compound predictor.

Users can use the following commands to configure the environment:
conda install scikit-learn=0.19.2
conda install -c openbabel openbabel


Running predictive scripts
python step5_pre_anti-SARS-CoV-2_compound.py ../data/test.smi ../data/test_prediction_result.txt
