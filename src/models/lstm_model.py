from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from src.models.base import BaseForecaster
from src.utils.logger import logger


class _LSTMNet(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class LSTMForecaster(BaseForecaster):
    def __init__(self, config: dict) -> None:
        super().__init__("lstm", config)
        self._scaler = MinMaxScaler()
        self._seq_len: int = config.get("lstm", {}).get("sequence_length", 30)
        self._last_seq: np.ndarray | None = None
        self._last_date: pd.Timestamp | None = None

    def _make_sequences(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self._seq_len, len(values)):
            X.append(values[i - self._seq_len : i])
            y.append(values[i])
        return np.array(X), np.array(y)

    def fit(self, train_data: pd.DataFrame, target_col: str = "total") -> None:
        cfg = self.config.get("lstm", {})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Aggregate across states → one value per date (LSTM takes a single series)
        if "date" in train_data.columns and "state" in train_data.columns and train_data["state"].nunique() > 1:
            agg = train_data.groupby("date")[target_col].sum().sort_index()
            series = agg.values.reshape(-1, 1)
            self._last_date = pd.Timestamp(agg.index[-1])
        else:
            series = train_data[target_col].values.reshape(-1, 1)
            if isinstance(train_data.index, pd.DatetimeIndex):
                self._last_date = train_data.index[-1]
            elif "date" in train_data.columns:
                self._last_date = pd.Timestamp(train_data["date"].iloc[-1])
            else:
                self._last_date = None

        scaled = self._scaler.fit_transform(series).flatten()

        X, y = self._make_sequences(scaled)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

        hidden = cfg.get("hidden_size", 64)
        layers = cfg.get("num_layers", 2)
        dropout = cfg.get("dropout", 0.2)
        epochs = cfg.get("epochs", 50)
        lr = cfg.get("learning_rate", 0.001)
        batch = cfg.get("batch_size", 32)

        net = _LSTMNet(
            input_size=1, hidden_size=hidden, num_layers=layers, dropout=dropout
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True)

        logger.info("LSTM training", device=str(device), epochs=epochs)
        net.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(
                    "LSTM epoch",
                    epoch=epoch + 1,
                    loss=round(epoch_loss / len(loader), 5),
                )

        self.model = net.cpu()  # move back to CPU for inference/save portability
        self._last_seq = scaled[-self._seq_len :]
        self.is_fitted = True
        logger.info("LSTM fitted", epochs=epochs, hidden=hidden, device=str(device))

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        cfg = self.config.get("lstm", {})
        mc_passes = cfg.get("mc_passes", 100)

        # Enable dropout at inference for MC Dropout CI
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.LSTM):
                module.train(False)

        seq = self._last_seq.copy()
        all_preds = []

        for _ in range(mc_passes):
            local_seq = seq.copy()
            pass_preds = []
            for _ in range(horizon):
                x = (
                    torch.tensor(local_seq[-self._seq_len :], dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                with torch.no_grad():
                    p = self.model(x).item()
                pass_preds.append(p)
                local_seq = np.append(local_seq, p)
            all_preds.append(pass_preds)

        all_preds = np.array(all_preds)  # (mc_passes, horizon)
        mean_scaled = all_preds.mean(axis=0)
        lower_scaled = np.percentile(all_preds, 2.5, axis=0)
        upper_scaled = np.percentile(all_preds, 97.5, axis=0)

        def inv(arr):
            return self._scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

        mean_val = np.maximum(inv(mean_scaled), 0)
        lower_val = np.maximum(inv(lower_scaled), 0)
        upper_val = np.maximum(inv(upper_scaled), 0)

        if hasattr(self, "_last_date") and self._last_date is not None:
            start = self._last_date + pd.offsets.Week(weekday=0)
        else:
            start = pd.Timestamp("today")
        dates = pd.date_range(start=start, periods=horizon, freq="W-MON")

        return pd.DataFrame(
            {
                "date": dates,
                "predicted_value": mean_val,
                "lower_bound": lower_val,
                "upper_bound": upper_val,
            }
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path + ".pt")
        joblib.dump(
            {
                "scaler": self._scaler,
                "last_seq": self._last_seq,
                "seq_len": self._seq_len,
                "model_cfg": self.config.get("lstm", {}),
                "last_date": getattr(self, "_last_date", None),
            },
            path + ".meta",
        )
        logger.info("LSTM saved", path=path)

    def load(self, path: str) -> None:
        meta = joblib.load(path + ".meta")
        self._scaler = meta["scaler"]
        self._last_seq = meta["last_seq"]
        self._seq_len = meta["seq_len"]
        self._last_date = meta.get("last_date")
        cfg = meta["model_cfg"]
        net = _LSTMNet(
            input_size=1,
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        )
        net.load_state_dict(torch.load(path + ".pt", weights_only=True))
        net.eval()
        self.model = net
        self.is_fitted = True
        logger.info("LSTM loaded", path=path)
