# Reproducibility repository for the paper *G-Mixup: Graph Data Augmentation for Graph Classification*

Original paper: https://arxiv.org/pdf/2202.07179.pdf

Original repository: https://github.com/ahxt/g-mixup

The repository is organized into several folders:
  - /fig/ contains all the figures generated during the reproducibility study and experimenting
  - /results/ contains logs for experiments that involve training GNNs
  - /src/ contains source code for the augmentation methods, models and training pipelines
    - Experiments 1, 2, 3, 6.1, 6.2, 7 and 8 have 3 files each associated with them: 
      - Jupyter notebook with the code needed to run the experiment, 
      - .py file with the notebook rewritten as a script that can be ran through terminal
    - Additionally, experiments 3, 6.1, 6.2 and 7 also have an accompanying Jupyter notebook that analyses the results of the experiment
    - models.py contains models used for experiments, augmentations.py contains all the augmentations, graphon_estimator.py contains all the graphon estimators, utils.py contains additional functions, gmixup.py is the file by original authors that runs an example of G-Mixup training. 
  - run_gmixup.sh and run_vanilla.sh are scripts by original authors with which you can run simple examples
  - requirements.txt contains all the neccessary packages. You can install them simply by running: 
  
  `$ pip install -r requirements.txt`
  
  
All the experiments were ran on Windows 10 OS, using torch 1.13.1 either with CPU or with 11.7 CUDA compilation tools.

- Ambiguities:
  - Nothing is written about how other augmentations were implemented. In case of DropEdge, DropNode and Subgraph augmentations, what percentage of the training batch is corrupted, and what is the ratio of corruption? Same goes for M-Manifold, what is the percentage of augmented graphs in the new training dataset?
  - Model hyperparameters: is dropout used during training? For GCN, hyperlinked is an example that used GCN2Conv layers. Are parameters \alpha and \theta kept as in example (\alpha=0.5, \theta=1.0), and do the layers share weights or not? For DiffPool and MinCutPool models, what is the parameter $max_nodes$ set to?
