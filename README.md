# Synthesizability-stoi-CGNF
Synthesizability-stoi-CGNF is a python code for predicting synthesizability score which is quantitative synthesizability metric of inorganic crystal compositions. This is a partially supervised machine learning protocol (PU-learning) using CGNF(Composition Graph Neural Fingerprint) atomic embedding method developed by prof. Yousung Jung group (contact: yousung.jung@snu.ac.kr).

## Developers
Jidon Jang, Juhwan Noh<br>

## Prerequisites
Python3<br> Numpy<br> Pytorch<br> Pymatgen<br>

## Publication
Jidon Jang, Juhwan Noh, Lan Zhou, Geun Ho Gu, John M. Gregoire, and Yousung Jung, "Predicting the Synthesizability of Materials Stoichiometry" (under review)

## Usage
### [1] Define a customized data format and prepare atomic embedding vector file for generation of CGNF
To input crystal structures to Synthesizability-stoi-CGNF, you will need to define a customized dataset and pre-generate CGNF as pickle files for bootstrap aggregating in semi-supervised learning. Note that this is required for both training and predicting.
Following files should be needed to generate CGNF.
#### id_prop.csv: a CSV file with two columns for positive data(synthesizable) and unlabeled data(not-yet-synthesized). The first column recodes a inorganic composition (The formula string format of Composition class in Pymatgen package is recommended), and the second column recodes the value (1 = positive, 0 = unlabeled) according to whether they were synthesized already or not.
#### cgcnn_hd_rcut4_nn8.element_embedding.json: a JSON file containing atomic embedding vectors for generation of CGNF

### [2] Train a Synthesizability-PU-CGCNN model
`python main_PU_learning.py --bag 100 --data id_prop.csv --embedding cgcnn_hd_rcut4_nn8.element_embedding.json --split ./split`<br>

Load composition information from 'id_prop.csv' and generate data split files for PU-learning in 'split' folder.<br>
After training, prediction results for test-unlabeled data (csv file) corresponding to each iteration will be generated.<br>
Result of bootstrap aggregating is saved as 'test_results_ensemble_100models.csv'<br>
You can change the number of bootstrap samples using '--bag' option<br>

### [3] Predict synthesizability of new crystals with pre-trained models
`python predict_PU_learning.py --bag 100 --data id_prop_test.csv --embedding cgcnn_hd_rcut4_nn8.element_embedding.json --modeldir ./models`<br>

Load composition information from 'id_prop_test.csv' file for test materials and pre-trained models from 'models' folder.<br>
Predict synthesizability of crystal composition in id_prop_test.csv file using the loaded models.<br>
Result of bootstrap aggregating is saved as 'test_results_ensemble_100models.csv'
