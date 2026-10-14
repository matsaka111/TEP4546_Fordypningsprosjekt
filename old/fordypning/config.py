# fordypning/config.py
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    n_species: int = 53          # GRI-Mech 3.0
    # input: T, T0, P, dt, Y (n_species)
    input_dim: int = 53 + 4
    # output: predicted Y and T at t+dt
    output_dim: int = 53 + 1
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

@dataclass
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    num_epochs: int = 200
    val_fraction: float = 0.2
    seed: int = 0

@dataclass
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 3

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    weights_dir: Path = Path("model_weights")