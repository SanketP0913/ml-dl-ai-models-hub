# ML · DL · AI Models Hub

A map of Artificial Intelligence, built one model at a time. The repo follows the AI family tree: **Machine Learning** and **Deep Learning** are the two main branches, each branch is split by type of learning, and every model gets its own folder with a real dataset, a runnable notebook and a README that explains it.

## Structure

```
Artificial Intelligence
├── 01-machine-learning/
│   ├── 01-supervised/         linear & logistic regression, decision tree, random forest, SVM, KNN, naive Bayes, XGBoost
│   ├── 02-unsupervised/       k-means, hierarchical clustering, DBSCAN, PCA, Gaussian mixture
│   ├── 03-semi-supervised/    self-training, label propagation
│   ├── 04-reinforcement/      Q-learning, SARSA, deep Q-network
│   └── 05-online-learning/    online SGD, passive-aggressive, Hoeffding tree
└── 02-deep-learning/
    ├── 01-cnn/                LeNet, VGG, ResNet
    ├── 02-rnn/                simple RNN
    ├── 03-lstm/               text sentiment, time series
    ├── 04-gru/                text classification
    ├── 05-transformer/        from scratch, BERT fine-tuning
    └── 06-autoencoder/        basic, denoising, VAE
```

## What is inside every model folder

- `README.md`: what the model is, which dataset it uses, why it fits, and how the results are measured (accuracy, F1, ROC AUC, RMSE and so on)
- `notebook.ipynb` or `model.py`: runs the model end to end
- `data/`: rebuilt by a download step, never stored in git

## Progress

| Model | Branch | Dataset | Status |
|---|---|---|---|
| [Loan default at approval time](01-machine-learning/01-supervised/lending-club-default) | Supervised | Lending Club, 2.26M loans | Done: Logistic Regression best, test ROC AUC 0.69 |

## Author

**Sanket Patil**, MPS Data Science, UMBC · [GitHub](https://github.com/SanketP0913)
