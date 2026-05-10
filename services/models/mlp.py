import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers=(128, 64),
        activation="relu",
        dropout_rate=0.2,
        use_batch_norm=False,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError("activation должен быть: 'relu', 'tanh', 'gelu', 'elu'")

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def make_features(series, lags, rolling_windows):
    df = pd.DataFrame({"y": series})

    for lag in lags:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for window in rolling_windows:
        shifted = df["y"].shift(1)
        rolling = shifted.rolling(window)
        df[f"rolling_mean_{window}"] = rolling.mean()
        df[f"rolling_std_{window}"] = rolling.std()
        df[f"rolling_min_{window}"] = rolling.min()
        df[f"rolling_max_{window}"] = rolling.max()

    return df.dropna()


def mlp_forecast_torch(
    series,
    horizon,
    lags=(1, 2, 3, 7, 14, 28),
    rolling_windows=(7, 14, 28),
    hidden_layers=(128, 64),
    activation="relu",
    dropout_rate=0.2,
    use_batch_norm=False,
    optimizer_name="adam",
    learning_rate=0.001,
    weight_decay=0.0,
    use_scheduler=True,
    scheduler_patience=5,
    scheduler_factor=0.5,
    epochs=30,
    batch_size=32,
    early_stopping=True,
    patience=5,
    scale_features=True,
    scale_target=False,
    device=None,
    random_state=42,
    verbose=False,
):
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = make_features(series, lags, rolling_windows)

    if df.empty:
        raise ValueError("Недостаточно данных для MLP")

    X = df.drop(columns=["y"]).values.astype(np.float32)
    y = df["y"].values.astype(np.float32)

    X_scaler = StandardScaler()
    if scale_features:
        X = X_scaler.fit_transform(X)

    y_scaler = None
    if scale_target:
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    X_tensor = torch.tensor(X.tolist(), dtype=torch.float32)
    y_tensor = torch.tensor(y.tolist(), dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(
        input_dim=X.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
    ).to(device)

    criterion = nn.MSELoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("optimizer_name должен быть: 'adam', 'adamw', 'sgd'")

    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    best_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        if scheduler is not None:
            scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.6f}")

        if early_stopping and patience_counter >= patience:
            if verbose:
                print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    history = list(series)
    preds = []
    feature_order = list(df.drop(columns=["y"]).columns)

    with torch.no_grad():
        for _ in range(horizon):
            features = {}

            for lag in lags:
                features[f"lag_{lag}"] = history[-lag] if lag <= len(history) else history[0]

            for window in rolling_windows:
                values = history[-window:] if len(history) >= window else history
                features[f"rolling_mean_{window}"] = np.mean(values)
                features[f"rolling_std_{window}"] = np.std(values)
                features[f"rolling_min_{window}"] = np.min(values)
                features[f"rolling_max_{window}"] = np.max(values)

            X_pred = pd.DataFrame(
                [[features[col] for col in feature_order]],
                columns=feature_order,
            ).values.astype(np.float32)

            if scale_features:
                X_pred = X_scaler.transform(X_pred)

            X_pred = torch.tensor(X_pred.tolist(), dtype=torch.float32).to(device)
            pred = model(X_pred).cpu().item()

            if y_scaler is not None:
                pred = y_scaler.inverse_transform([[pred]])[0][0]

            preds.append(pred)
            history.append(pred)

    return np.array(preds), model
