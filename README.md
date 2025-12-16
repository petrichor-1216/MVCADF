MVCADF: Multi-View Contrastive Alignment for Drug-Drug Interaction Prediction

MVCADF is a deep learning framework designed to predict drug-drug interactions (DDI) using multi-view drug representations. The model leverages multiple views of drug information, including molecular structures, node features, motif graphs, element graphs, and drug fingerprints. It combines these views using dynamic fusion and cross-view contrastive alignment to generate robust drug representations for DDI prediction.

Features

Multi-View Fusion: Integrates various views such as molecular graphs, node features, motifs, element graphs, and fingerprints.

Graph Neural Networks (GNN): Uses various GNN layers like Graph Convolutional Networks (GCN) and Graph Transformers to capture node and graph-level representations.

Contrastive Learning: Applies contrastive losses (e.g., multi-view alignment loss and mutual information loss) to encourage consistency across views.

Dynamic Fusion: Adapts the fusion of multi-view representations to highlight the most informative features.

End-to-End Framework: End-to-end process including data preprocessing, feature extraction, graph representation, and DDI prediction.

Requirements

This framework requires the following dependencies:

Dependencies

Python 3.9.x

PyTorch 2.0.1+cu117

PyTorch Geometric 2.0.2

RDKit 2022.9.4

scikit-learn 1.4.2

tqdm 4.66.5

Matplotlib 3.9.4

NumPy 1.24.4

NetworkX 2.8.8

DGL 2.1.0+cu117

You can install the required packages by using conda or pip. To install the environment from the provided environment.yml or a similar setup:

conda create --name Tiger python=3.9
conda activate Tiger
conda install pytorch torchvision torchaudio pytorch-geometric rdkit scikit-learn tqdm matplotlib networkx dgl -c conda-forge


Alternatively, install with pip:

pip install torch torchvision torchaudio torch-geometric rdkit scikit-learn tqdm matplotlib networkx dgl

Model Overview
Components

Graph Transformer: Uses Graph Transformer layers to model graph-level dependencies and perform drug feature extraction.

Multi-View Feature Encoders: Encodes drug information into multiple views such as molecular graphs, motifs, fingerprints, and element graphs.

Dynamic Fusion: The DynamicFuser dynamically combines the views, learning which views to prioritize during prediction.

Contrastive Learning: The model aligns different views using a pair-wise alignment loss and a multi-view center alignment loss.

Discriminator: A discriminator is used for mutual information maximization between the various drug views.

Model Workflow

Input: The model takes in drug-related data from different sources (e.g., SMILES, motifs, element graphs, fingerprints).

Feature Extraction: Various encoders process the drug data and extract features from the corresponding views.

Fusion: Dynamic fusion of the features occurs using the DynamicFuser, which computes weighted fusion of the multi-view representations.

Prediction: The fused features are projected and classified using fully connected layers.

Loss Function

The model is trained with a combination of:

Label Loss: Cross-entropy loss on the predicted drug-drug interaction (DDI) labels.

Contrastive Losses: For enforcing consistency between multiple drug representations.

Multi-View Alignment Loss: To align the drug representations across different views.

Mutual Information Loss: To maximize the mutual information between the features of different views.

Usage
1. Data Preparation

Before training, ensure the following files are available in the dataset:

drug_smiles.txt: Contains the SMILES strings of the drugs.

networks.txt: Defines the drug-drug interaction network structure.

ddi.txt: Contains DDI pairs and their labels.

element_graph.pt: Element graph data for drug representations.

motif_graph.pt: Motif graph data for drug representations.

2. Running the Model

To run the MVCADF model, use the following command:

python main.py --model_name mvcadf --dataset drugbank --folds 5 --batch_size 128


This will:

Initialize the MVCADF model with the specified parameters.

Train the model using 5-fold cross-validation.

Output evaluation metrics such as accuracy, F1-score, AUC, and AUPR.

3. Model Evaluation

The model computes the following metrics during training and evaluation:

Accuracy: The percentage of correctly predicted DDI pairs.

F1-Score: A harmonic mean of precision and recall.

AUC (Area Under ROC Curve): A measure of the model's ability to rank DDI pairs.

AUPR (Area Under Precision-Recall Curve): A measure of the model's ability to identify true positives in DDI predictions.

4. Saving and Loading the Model

After training, the model can be saved using:

model.save('path_to_save')


To load the model:

model.load_state_dict(torch.load('path_to_saved_model'))

Results

The framework evaluates the model on various metrics and saves the best model after each fold. Metrics such as accuracy, F1-score, AUC, and AUPR are logged for both training and evaluation sets.

Contact

For any questions or issues, please open an issue in the repository.
