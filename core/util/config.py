
from dataclasses import dataclass


@dataclass
class NCTModelConfig:
    dataset_normalized: bool
    gnn_model_type: str
    gnn_hidden_size: int
    gnn_activation_fn: str
    mlp_hidden_size: int
    mlp_activation_fn: str
    num_gat_heads: int
    num_message_passing: int
    ablation_pc: bool
    ablation_midi: bool
    ablation_beat: bool
    ablation_edge_feat: bool
    dropout_p: float


@dataclass
class TrainConfig:
    dir_preprocessed_dataset: str
    debug: bool
    device: str
    seed: int
    cv_set_no: int
    batch_size: int
    learning_rate: float
    metric: str
    fscore_beta: float
    warmup_nct: int
    gradient_clip_value: float
    patience: int
