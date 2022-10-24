# Digital Asset Valuation: A Study on Domain Names, Email Addresses, and NFTs

This repository hosts the DASH dataset and the associated code.

DASH is the first **D**igital **A**sset **S**ales **H**istory dataset that features multiple digital asset classes spanning from classical to blockchain-based ones, consisting of 280K transactions of domain names, email addresses, and non-fungible token (NFT)-based identifiers.

We also propose several valuation models for DASH, including conventional feature-based models and deep learning models, all applicable to multiple asset classes, and we, for the first time, demonstrate that fine-tuning a pre-trained model can beat conventional models in digital asset valuation.

If you find this work useful, please cite the following article [(arXiv)](https://arxiv.org/abs/2210.10637).

```
@article{sun2022digital,
  doi = {10.48550/ARXIV.2210.10637},
  url = {https://arxiv.org/abs/2210.10637},
  author = {Sun, Kai},
  title = {Digital Asset Valuation: A Study on Domain Names, Email Addresses, and NFTs},
  journal = {arXiv preprint},
  volume = {cs.IR/2210.10637},
  year = {2022}
}
```

Files in this repository:

* ```LICENSE```: the license of the DASH dataset and the associated code.
* ```data/v1.0/dash_{dn,ea,nft}.json```: the DASH dataset.
* ```conventional.py```: the code for training and evaluating conventional models. The code has been tested with Python 3.7.1, Scikit-learn 0.20.2, and XGBoost 1.5.1. 
* ```neural.ipynb```: a Colab adaptation of vanilla mBERT and mBERT+. Note that the code is non-deterministic due to GPU non-determinism. 
* ```features.json```: pre-extracted features for assets in DASH, including vocabulary, number of tokens, trademark, and TLD count.
* ```misc/adult_keywords```: the adult word and phrase list used in the paper.


