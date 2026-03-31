# ML4SCI GSoC 2026 — Non-local GNNs for Jet Classification

Evaluation tasks and proposal for GSoC 2026 under ML4SCI (GENIE project).

## Evaluation Tasks

**Common Task 1: Autoencoder** (`common_task_1.ipynb`)
- Fully convolutional autoencoder for quark/gluon jet image reconstruction
- Log transform + weighted MSE loss to handle extreme sparsity
- Trained on 10K events, includes residual analysis and latent space visualization

**Common Task 2: GNN Classification** (`common_task_2.ipynb`)
- EdgeConv GNN for quark/gluon binary classification
- Images converted to point clouds with k=7 nearest neighbor graphs
- Test AUC: 0.788, Accuracy: 72%

**Specific Task 4: Non-local GNNs** (`specific_task_4.ipynb`)
- Comparison of three architectures: Baseline EdgeConv, EdgeConv + Self-Attention, Graph Transformer
- All models trained on identical data and settings
- Graph Transformer achieved highest AUC (0.794)

## Proposal

Full proposal available in `ML4SCI_Proposal.pdf`

## Requirements

- PyTorch
- PyTorch Geometric
- h5py, scikit-learn, matplotlib
