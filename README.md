#  Housing Price Prediction with PyTorch

## Overview
This project implements a neural network using PyTorch to predict apartment prices in Jordan based on property features. The model learns the relationship between input features and the target variable through supervised learning.

## What the Model Predicts
The model predicts the **apartment price (`price_jod`)** in Jordanian Dinars.

Input Features:
- `area_sqm` — Apartment area in square meters
- `bedrooms` — Number of bedrooms
- `floor` — Floor number
- `age_years` — Age of the building in years
- `distance_to_center_km` — Distance to city center in kilometers

## Setup
python -m venv .venv
.venv\Scripts\activate
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python train.py

## Model Architecture
Linear(5 → 32) → ReLU → Linear(32 → 1)

## Training Configuration
Epochs: 100
Learning Rate: 0.01
Optimizer: Adam
Loss Function: Mean Squared Error (MSELoss)

## Training Outcome
Initial Loss (Epoch 0): 1950600192.0000
Final Loss (Epoch 90): 1944283392.0000
The loss decreased during training, indicating that the model successfully learned from the data.

## Behavioral Observation
The loss decreased quickly during the first 20 epochs, then gradually stabilized. This shows the model captured main patterns first, then refined predictions slowly.

## Project Structure
├── train.py              ← Training script (model + training loop)
├── data/
│   └── housing.csv       ← Dataset
├── predictions.csv       ← Model predictions (generated after training)
├── README.md             ← Required: setup, overview, project structure
├── loss_curve.png        ← Loss vs epochs
├── prediction_plot.png   ← actual valus VS predected Value

## Output
predictions.csv contains:
actual — True prices
predicted — Model predictions

## Deactivate Virtual Environment
deactivate


## Overfitting Discussion

After training the Housing Price Prediction model, we compared the metrics on the training and test sets to evaluate performance:

Metric	Train	Test
MAE	6114.55 JOD	5313.50 JOD
R²	0.7280	0.6615
Analysis
Train vs. Test Performance:
The training and test MAE values are fairly close, and the R² scores are reasonably high on both sets. This suggests that the model is not severely overfitting — it generalizes fairly well to unseen data.
Visual Inspection:
The scatter plot of predicted vs. actual prices shows most points near the line of perfect prediction, supporting the numerical results.

Next Steps
To further improve the model and prevent potential overfitting in more complex scenarios:

Try more advanced architectures:
Add more layers or neurons.
Experiment with activation functions or dropout.

Hyperparameter Tuning:
Adjust learning rate schedule.
Experiment with batch sizes or optimizers.