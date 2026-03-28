# """
# Integration 2 — PyTorch: Housing Price Prediction
# Module 2 — Programming for AI & Data Science

# Complete each section below. Remove the TODO: comments and pass statements
# as you implement each section. Do not change the overall structure.

# Before running this script, install PyTorch:
#     pip install torch --index-url https://download.pytorch.org/whl/cpu
# """

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch

# print(torch.__version__)
# print(torch.rand(3, 3))

# # ─── Model Definition ─────────────────────────────────────────────────────────


# class HousingModel(nn.Module):
#     """Neural network for predicting housing prices from property features.

#     Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
#     """

#     def __init__(self):
#         """Define the model layers."""
#         super().__init__()

#         self.layer1 = nn.Linear(5, 32)  # — 5 input features → 32 hidden units
#         self.relu = nn.ReLU()  # — activation function
#         self.layer2 = nn.Linear(32, 1)  # — 32 hidden → 1 output (price prediction)
#         pass

#     def forward(self, x):
#         """Define the forward pass.

#         Args:
#             x (torch.Tensor): Input tensor of shape (N, 5).

#         Returns:
#             torch.Tensor: Predictions of shape (N, 1).
#         """

#         x = self.layer1(x)
#         x = self.relu(x)
#         x = self.layer2(x)
#         return x
#         pass


# # ─── Main Training Script ─────────────────────────────────────────────────────


# def main():
#     """Load data, train HousingModel, and save predictions."""

#     # ── 1. Load Data ──────────────────────────────────────────────────────────
#     df = pd.read_csv("data/housing.csv")
#     print("Data shape:", df.shape)

#     # ── 2. Separate Features and Target ──────────────────────────────────────
#     feature_cols = [
#         "area_sqm",
#         "bedrooms",
#         "floor",
#         "age_years",
#         "distance_to_center_km",
#     ]
#     X = df[feature_cols]
#     y = df[["price_jod"]]

#     # ── 3. Standardize Features ───────────────────────────────────────────────
#     X_mean = X.mean()
#     X_std = X.std()
#     X_scaled = (X - X_mean) / X_std

#     y_mean = y.mean()
#     y_std = y.std()
#     y_scaled = (y - y_mean) / y_std
#     # Why: features have very different scales; standardization ensures
#     #      gradient updates are balanced across all input dimensions.
#     split_par = int(len(X_scaled) * 0.8)
#     x_train = X_scaled[:split_par]
#     x_Test = X_scaled[split_par:]
#     y_train, y_Test = y_scaled[:split_par], y_scaled[split_par:]
#     print(x_train.shape, y_train.shape)

#     def mae(y, pred):
#         Mae_test = np.mean(np.abs((y - pred.detach().numpy())))
#         return Mae_test

#     def r2(y, pred):
#         ss_res = np.sum((y - pred) ** 2)
#         ss_tot = np.sum((y - np.mean(y)) ** 2)
#         return 1 - ss_res / ss_tot

#     # ── 4. Convert to Tensors ─────────────────────────────────────────────────
#     X_tensor = torch.tensor(x_train.values, dtype=torch.float32)
#     y_tensor = torch.tensor(y_train.values, dtype=torch.float32)
#     X_test_tensor = torch.tensor(x_Test.values, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_Test.values, dtype=torch.float32)

#     print("X shape:", X_tensor.shape)
#     print("y shape:", y_tensor.shape)

#     # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────

#     model = HousingModel()
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     # ── 6. Training Loop ──────────────────────────────────────────────────────
#     num_epochs = 100

#     for epoch in range(num_epochs):

#         model.train()
#         predictions = model(X_tensor)

#         loss = criterion(predictions, y_tensor)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if epoch % 10 == 0:
#             print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")
#             print(f"mae_def = {mae(y= y_train,pred= predictions)}")

#     # ── 7. Save Predictions ───────────────────────────────────────────────────
#     with torch.no_grad():
#         predictions_tensor = model(X_tensor)
#         y_pred_test = model(X_test_tensor).numpy().flatten()
#         y_actual_test = y_test_tensor.numpy().flatten()

#     predictions_np = predictions_tensor.numpy().flatten()
#     actuals_np = y_tensor.numpy().flatten()
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ax.scatter(
#         y_actual_test, y_pred_test, color="blue", alpha=0.6, label="Predicted vs Actual"
#     )

#     # Line of perfect prediction
#     min_val = min(y_actual_test.min(), y_pred_test.min())
#     max_val = max(y_actual_test.max(), y_pred_test.max())
#     ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

#     ax.set_xlabel("Actual Price (JOD)")
#     ax.set_ylabel("Predicted Price (JOD)")
#     ax.set_title("Housing Price Prediction — Test Set")
#     ax.legend()
#     plt.savefig("predictions_plot.png")
#     plt.show()
#     plt.close(fig)
#     results_df = pd.DataFrame({"actual": actuals_np, "predicted": predictions_np})

#     results_df.to_csv("predictions.csv", index=False)
#     print("Saved predictions.csv")


# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# ─── Model Definition ─────────────────────────────────────────────────────────
class HousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 32)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Main Script ─────────────────────────────────────────────────────────────
def main():
    # 1. Load Data
    df = pd.read_csv("data/housing.csv")
    print("Data shape:", df.shape)

    # 2. Features & Target
    feature_cols = [
        "area_sqm",
        "bedrooms",
        "floor",
        "age_years",
        "distance_to_center_km",
    ]

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df[feature_cols]
    y = df[["price_jod"]]

    # ─── Scaling ─────────────────────────────────────────────────────────────
    X_mean, X_std = X.mean(), X.std()
    X_scaled = (X - X_mean) / X_std

    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std

    # 4. Train/Test Split
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    print("X_train shape:", X_train_tensor.shape)

    # ─── Model ─────────────────────────────────────────────────────────────
    model = HousingModel()
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ─── Training ───────────────────────────────────────────────────────────
    num_epochs = 100
    losses = []

    for epoch in range(num_epochs):
        model.train()

        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 10 == 0:

            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ─── Evaluation ─────────────────────────────────────────────────────────
    with torch.no_grad():
        y_train_pred = model(X_train_tensor).numpy()
        y_test_pred = model(X_test_tensor).numpy()

    y_train_pred = y_train_pred * y_std.values + y_mean.values
    y_test_pred = y_test_pred * y_std.values + y_mean.values

    y_train_np = y.values[:split_idx].flatten()
    y_test_np = y.values[split_idx:].flatten()

    y_train_pred = y_train_pred.flatten()
    y_test_pred = y_test_pred.flatten()

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    print(
        f"Train MAE: {mae(y_train_np, y_train_pred):.2f}, R²: {r2(y_train_np, y_train_pred):.4f}"
    )
    print(
        f"Test  MAE: {mae(y_test_np, y_test_pred):.2f}, R²: {r2(y_test_np, y_test_pred):.4f}"
    )

    # ─── Plot Predictions ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    errors = np.abs(y_test_pred - y_test_np)

    scatter = ax.scatter(
        y_test_np, y_test_pred, c=errors, cmap="viridis", s=60, alpha=0.7
    )

    min_val = min(y_test_np.min(), y_test_pred.min())
    max_val = max(y_test_np.max(), y_test_pred.max())

    ax.plot([min_val, max_val], [min_val, max_val], "r--")

    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Predictions")

    plt.colorbar(scatter, ax=ax)
    plt.show()
    plt.close(fig)
    # ─── Loss Curve ─────────────────────────────────────────────────────────
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
