import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class RNNForecastModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        rnn_type="lstm",
        bidirectional=False,
    ):
        super().__init__()

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("rnn_type должен быть 'lstm' или 'gru'")

        direction_multiplier = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * direction_multiplier, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)


def rnn_forecast_torch(
    series,
    horizon,
    sequence_length=28,
    rnn_type="lstm",
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    bidirectional=False,
    optimizer_name="adamw",
    learning_rate=0.001,
    weight_decay=1e-4,
    use_scheduler=True,
    scheduler_patience=5,
    scheduler_factor=0.5,
    epochs=30,
    batch_size=32,
    early_stopping=True,
    patience=5,
    scale_series=True,
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

    values = np.array(series).reshape(-1, 1)

    if len(values) <= sequence_length:
        raise ValueError("Недостаточно данных для RNN")

    scaler = StandardScaler()
    if scale_series:
        values = scaler.fit_transform(values)

    values = values.flatten()

    X = []
    y = []
    for i in range(sequence_length, len(values)):
        X.append(values[i - sequence_length:i])
        y.append(values[i])

    X = np.expand_dims(np.array(X).astype(np.float32), axis=-1)
    y = np.array(y).astype(np.float32)

    X_tensor = torch.tensor(X.tolist(), dtype=torch.float32)
    y_tensor = torch.tensor(y.tolist(), dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RNNForecastModel(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
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
        raise ValueError("optimizer_name должен быть 'adam', 'adamw', 'sgd'")

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
    history = list(values)
    preds = []

    with torch.no_grad():
        for _ in range(horizon):
            seq = np.array(history[-sequence_length:]).astype(np.float32)
            seq = np.expand_dims(seq, axis=0)
            seq = np.expand_dims(seq, axis=-1)

            X_pred = torch.tensor(seq.tolist(), dtype=torch.float32).to(device)
            pred = model(X_pred).cpu().item()
            history.append(pred)

            if scale_series:
                pred = scaler.inverse_transform([[pred]])[0][0]

            preds.append(pred)

    return np.array(preds), model
