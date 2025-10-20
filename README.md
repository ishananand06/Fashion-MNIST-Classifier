# Fashion-MNIST Classification with PyTorch

This project is a hands-on exploration of building and optimising a neural network to classify clothing items from the Fashion-MNIST dataset.

The primary goal was to systematically improve a simple neural network by tuning its architecture and hyperparameters.

## Final Results

The best-performing model was a **2-layer network (256 -> 128 neurons)** using the **Adam optimiser**, which achieved a peak accuracy of **89.0%**.

## Experiments & Learnings

This project was built through a series of iterative experiments.

#### 1. Baseline Model (`layer1-128.py`)
* **Architecture:** 1 Hidden Layer (128 neurons)
* **Optimizer:** `Adam(lr=0.001)`
* **Result:** ~88.3% Accuracy.
* **Learning:** The Adam optimiser provides a very strong and fast baseline, converging to a high accuracy in just a few epochs.

#### 2. Tuning Network "Width" (`layer1-256.py`)
* **Architecture:** 1 Hidden Layer (256 neurons)
* **Optimizer:** `Adam(lr=0.001)`
* **Result:** ~88.7% Accuracy.
* **Learning:** A "wider" network (more neurons) provided a minor performance improvement.

#### 3. Tuning Network "Depth" (`layer2-*.py`)
Two 2-layer architectures were tested:
* `layer2-256-64.py`: (256 -> 64 neurons) -> **~88.5% Accuracy.**
* `layer2-256-128.py`: (256 -> 128 neurons) -> **~89.0% Accuracy.**
* **Optimizer:** `Adam(lr=0.001)`
* **Learning:** A "deeper" network was more effective than just a "wider" one. The `256 -> 128` "funnel" architecture was the most effective, suggesting the `256 -> 64` model created too much of a bottleneck.

#### 4. Regularisation (`layer2-256-128+Dropout.py`)
* **Architecture:** 2 Layers (256 -> 128) with `Dropout(p=0.2)`
* **Optimizer:** `Adam(lr=0.001)`
* **Result:** ~88.5% Accuracy.
* **Learning:** Adding dropout made the model train slower and slightly reduced the 10-epoch accuracy. This suggests that for 10 epochs, the model wasn't overfitting significantly, and the main effect of dropout was just making the training harder.
