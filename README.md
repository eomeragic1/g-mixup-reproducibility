# Reproducibility report for the paper *G-Mixup: Graph Data Augmentation for Graph Classification*

- Current ambiguities:
  - Nothing is written about how other augmentations were implemented. In case of DropEdge, DropNode and Subgraph augmentations, what percentage of the training batch is corrupted, and what is the ratio of corruption? Same goes for M-Manifold, what is the percentage of augmented graphs in the new training dataset?
  - Model hyperparameters: is dropout used during training? For GCN, hyperlinked is an example that used GCN2Conv layers. Are parameters \alpha and \theta kept as in example (\alpha=0.5, \theta=1.0), and do the layers share weights or not? For DiffPool and MinCutPool models, what is the parameter $max_nodes$ set to?
