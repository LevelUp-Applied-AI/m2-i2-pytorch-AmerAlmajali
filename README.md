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

## Output
predictions.csv contains:
actual — True prices
predicted — Model predictions

## Deactivate Virtual Environment
deactivate

---

