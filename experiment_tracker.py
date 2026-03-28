import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json
import time


class HousingModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.tensor):
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)


def main():

    # Import and shuffle data

    df = pd.read_csv("data/housing.csv")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    feature_cols = [
        "area_sqm",
        "bedrooms",
        "floor",
        "age_years",
        "distance_to_center_km",
    ]
    X = df[feature_cols]
    y = df["price_jod"]

    # Standardization

    x_scaled = (X - X.mean()) / X.std()
    y_scaled = (y - y.mean()) / y.std()

    # Save mean/std for inverse scaling
    y_mean = y.mean()
    y_std = y.std()

    # Train/test split

    split_par = int(len(x_scaled) * 0.8)
    x_train_scaled, x_test_scaled = x_scaled[:split_par], x_scaled[split_par:]
    y_train_scaled, y_test_scaled = y_scaled[:split_par], y_scaled[split_par:]

    X_train_tensor = torch.tensor(x_train_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled.values, dtype=torch.float32).unsqueeze(
        1
    )

    X_test_tensor = torch.tensor(x_test_scaled.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled.values, dtype=torch.float32).unsqueeze(1)

    # Hyperparameters

    IR_list = [0.0001, 0.001, 0.01]
    Hdd_size = [16, 32, 64]
    epochs_list = [50, 100, 300]

    loss_fn = nn.MSELoss()
    experiments = []

    # Experiment loop

    for hidden_size in Hdd_size:
        for lr in IR_list:
            for epochs in epochs_list:
                start_time = time.time()

                model = HousingModel(hidden_size=hidden_size)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                print(f"\nHidden_size={hidden_size}, LR={lr}, Epochs={epochs}")

                for epoch in range(epochs):
                    model.train()
                    y_pred = model(X_train_tensor)
                    loss = loss_fn(y_pred, y_train_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch % 10 == 0:
                        print(f"Epoch={epoch}, Train Loss={loss.item():.4f}")

                # Evaluation

                model.eval()
                with torch.inference_mode():
                    y_pred_test = model(X_test_tensor)
                    loss_test = loss_fn(y_pred_test, y_test_tensor)

                # Convert to original scale
                y_pred_real = y_pred_test.numpy().flatten() * y_std + y_mean
                y_true_real = y_test_tensor.numpy().flatten() * y_std + y_mean

                # Metrics
                mae = np.mean(np.abs(y_true_real - y_pred_real))
                ss_res = np.sum((y_true_real - y_pred_real) ** 2)
                ss_tot = np.sum((y_true_real - np.mean(y_true_real)) ** 2)
                r2 = 1 - (ss_res / ss_tot)

                end_time = time.time()

                print(f"Test Loss={loss_test.item():.4f}, MAE={mae:.2f}, R2={r2:.4f}")

                # Save experiment
                experiments.append(
                    {
                        "hidden_size": hidden_size,
                        "lr": lr,
                        "epochs": epochs,
                        "train_loss": loss.item(),
                        "test_loss": loss_test.item(),
                        "mae": mae,
                        "r2": r2,
                        "time": end_time - start_time,
                    }
                )

    # Save experiments to JSON

    with open("experiments.json", "w") as f:
        json.dump(experiments, f, indent=4)

    # Leaderboard

    experiments_sorted = sorted(experiments, key=lambda x: x["mae"])

    print("\n========== TOP 10 EXPERIMENTS ==========")
    print(
        f"{'Rank':<5} | {'LR':<7} | {'Hidden':<6} | {'Epochs':<6} | {'Test MAE':<12} | {'Test R²':<9} | {'Time (s)':<8}"
    )
    for i, exp in enumerate(experiments_sorted[:10]):

        print(
            f"{i+1:<5} | "
            f"{exp['lr']:<7} | "
            f"{exp['hidden_size']:<6} | "
            f"{exp['epochs']:<6} | "
            f"{exp['mae']:<12.2f} | "
            f"{exp['r2']:<9.4f} | "
            f"{exp['time']:<8.2f}"
        )

    # Visualization

    plt.figure(figsize=(8, 5))
    markers = {50: "o", 100: "s", 300: "x"}

    for exp in experiments:
        lr = exp["lr"]
        mae = exp["mae"]
        hidden = exp["hidden_size"]
        epochs = exp["epochs"]

        plt.scatter(lr, mae, label=f"H{hidden}-E{epochs}", marker=markers[epochs], s=60)

    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("MAE (JOD)")
    plt.title("Experiment Summary")

    # Remove duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=7)

    plt.savefig("experiment_summary.png")
    print("\nSaved experiment_summary.png")


if __name__ == "__main__":
    main()
