# Copilot Instructions for Package Locker Forecasting Project

This is a machine learning research project focused on forecasting package withdrawal patterns at NYC smart lockers using time series data.

## Project Architecture

This project follows a **notebook-driven research workflow** with three main forecasting approaches:
- **LSTM.ipynb**: Basic LSTM implementation using PyTorch Lightning
- **FL-LSTM.ipynb**: Advanced LSTM with Federated Learning simulation
- **XGBoost.ipynb**: Tree-based baseline model for comparison

### Core Data Pipeline

All notebooks follow this standardized data flow:
1. Load from `dataset/locker_nyc_engineered.csv` (pre-aggregated hourly counts by locker)
2. Engineer time-based features: `hour_sin/cos`, `IsWeekend`, `IsPeakHour`  
3. Create rolling window features: `withdraw_1/2`, `delivery_1/2/8/16`
4. Calculate inventory proxy: `cumsum(delivery_1 - withdraw_1)`
5. Target: `proportion_withdraw = size_S_withdraw / inventory` (proportion, not raw counts)

### Key Patterns & Conventions

**Data Splitting**: Always use temporal splits, never random
```python
# Time-based split per notebook pattern
split_idx = int(len(df) * (1 - test_size))
train_df = df[:split_idx]
test_df = df[split_idx:]
```

**Model Configuration Constants** (consistent across LSTM notebooks):
```python
ENCODER_LENGTH = 16    # Input sequence length
DECODER_LENGTH = 4     # Prediction horizon
TARGET_COL = 'proportion_withdraw'
BATCH_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 8
```

**Custom Loss Function**: Use `DecayMSE(lambda_decay=0.95)` to weight recent predictions higher:
```python
class DecayMSE(nn.Module):
    # Applies exponential decay: w_h = exp(-λ * h)
```

**Feature Engineering Pipeline**:
- Drop columns with pattern: `[col for col in df.columns if col not in ['Date Hour', 'Locker Name', 'proportion_withdraw', 'time_idx', 'size_S_withdraw']]`
- Always use `StandardScaler` for LSTM features, sharing train scaler with val/test
- XGBoost uses raw features without scaling

## Development Workflow

**Environment**: UV package manager with PyTorch Lightning + forecasting stack
```bash
# Dependencies managed in pyproject.toml
# Key packages: pytorch-forecasting, pytorch-lightning, xgboost
```

**Training**: PyTorch Lightning with GPU acceleration
```python
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=500,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
)
```

**Evaluation Patterns**:
- Multi-horizon analysis: separate metrics for each prediction step (h=0,1,2,3)
- Residual analysis with studentized residuals and autocorrelation plots
- Convert proportion predictions back to counts: `predicted_withdraw = predictions * inventory`

## Data Characteristics

**Target Distribution**: Highly zero-inflated proportion_withdraw (~X% zeros)
**Temporal Granularity**: 3-hour resampled intervals (from original hourly)
**Key Features**: 
- `inventory` (proxy calculated from delivery-withdraw cumsum)
- Cyclical time encodings (`hour_sin/cos`)
- Rolling window aggregations
- Binary flags (`IsIndoor`, `IsWeekend`, `IsPeakHour`)

**Locker Types**: Mix of indoor/outdoor locations across NYC

## Model-Specific Notes

**LSTM Models**: Require careful sequence construction with `TimeSeriesDataset` class
**XGBoost**: Uses standard tabular approach, no sequence structure
**Evaluation**: Focus on multi-step forecasting accuracy, especially proportion→count conversion

## Critical Implementation Details

- Always validate shape matching: `assert y.shape == y_hat.shape`
- Use `sigmoid` activation for proportion outputs (0-1 bounds)
- Lightning logs stored in `lightning_logs/` with version tracking
- Model checkpoints: save with `torch.save(model.state_dict(), 'lstm_model.pth')`

This project prioritizes research experimentation over production deployment - focus on model performance analysis and feature engineering insights.