# GNN-RF Propagation V2
# Configuração Central do Projeto

"""
Configuração central baseada na estrutura real do dataset GNN_TOPO.
Validado contra RAG_CONTROL/annotated/data/graph_structure.md
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import torch


@dataclass
class TerrainConfig:
    """Configuração do terreno baseada na GNN_TOPO real."""
    
    # Features de entrada (17 dimensões) - DO RAG
    input_dim: int = 17
    feature_names: List[str] = field(default_factory=lambda: [
        'elevation',      # 0: SRTM/LiDAR, Z-score
        'slope',          # 1: [0,1], 0-90°
        'aspect_cos',     # 2: [-1,1]
        'aspect_sin',     # 3: [-1,1]
        'curvature',      # 4: raw
        'tpi',            # 5: Z-score
        'tri',            # 6: Z-score
        'roughness',      # 7: Z-score
        'b02_norm',       # 8: Sentinel-2 Blue [0,1]
        'b03_norm',       # 9: Sentinel-2 Green [0,1]
        'b04_norm',       # 10: Sentinel-2 Red [0,1]
        'b08_norm',       # 11: Sentinel-2 NIR [0,1]
        'ndvi',           # 12: [-1,1]
        'ndwi',           # 13: [-1,1]
        'has_lidar',      # 14: 0/1
        'z_lidar',        # 15: metros
        'confidence'      # 16: [0,1]
    ])
    
    # Targets da GNN_TOPO (14 dimensões) - DO RAG
    output_dim: int = 14
    target_names: List[str] = field(default_factory=lambda: [
        'elevation',      # 0
        'slope',          # 1
        'aspect_cos',     # 2
        'aspect_sin',     # 3
        'curvature',      # 4
        'tpi',            # 5
        'tri',            # 6
        'roughness',      # 7
        'flow_accum',     # 8
        'canopy',         # 9 - CRÍTICO para RF
        'ndvi',           # 10
        'ndwi',           # 11
        'bsi',            # 12
        'shadow'          # 13
    ])
    
    # Índice do canopy height - CRÍTICO
    canopy_index: int = 9
    
    # Edge features (2 dimensões) - DO RAG
    edge_dim: int = 2
    edge_feature_names: List[str] = field(default_factory=lambda: [
        'dist_norm',      # Distância normalizada [0,1]
        'dz_norm'         # Diferença de elevação Z-score
    ])


@dataclass
class AntennaConfig:
    """Configuração das antenas ANATEL."""
    
    # Features de entrada (10 dimensões)
    input_dim: int = 10
    feature_names: List[str] = field(default_factory=lambda: [
        'lat',              # 0: [-90, 90]
        'lon',              # 1: [-180, 180]
        'altura_antena',    # 2: [0, 200] metros
        'freq_tx_mhz',      # 3: [700, 6000] MHz
        'potencia_watts',   # 4: log-scaled
        'ganho_antena',     # 5: [0, 30] dBi
        'azimute_sin',      # 6: [-1, 1]
        'azimute_cos',      # 7: [-1, 1]
        'tilt',             # 8: [-20, 90] graus
        'tecnologia_enc'    # 9: encoding
    ])
    
    # Ranges de normalização
    freq_range: tuple = (700, 6000)       # MHz
    potencia_range: tuple = (0.1, 10000)  # Watts
    altura_range: tuple = (0, 200)        # metros
    ganho_range: tuple = (0, 30)          # dBi


@dataclass
class RFOutputConfig:
    """Configuração das saídas RF."""
    
    # Targets RF (5 dimensões)
    output_dim: int = 5
    target_names: List[str] = field(default_factory=lambda: [
        'path_loss_total',      # 0: dB
        'path_loss_vegetation', # 1: dB
        'path_loss_terrain',    # 2: dB
        'rssi',                 # 3: dBm
        'coverage_prob'         # 4: [0,1]
    ])
    
    # Ranges físicos
    path_loss_range: tuple = (0, 200)     # dB
    rssi_range: tuple = (-150, 0)         # dBm
    
    # Thresholds de cobertura (dBm)
    coverage_classes: Dict[int, tuple] = field(default_factory=lambda: {
        1: (-70, float('inf')),   # Excelente
        2: (-85, -70),            # Bom
        3: (-100, -85),           # Regular
        4: (-110, -100),          # Fraco
        5: (float('-inf'), -110)  # Sem cobertura
    })


@dataclass
class ModelConfig:
    """Configuração do modelo (compatível com TerrainGNN v21)."""
    
    # Dimensões (DO RAG - architecture_v21.md)
    hidden_dim: int = 512
    num_layers: int = 4      # Menos que TOPO (6)
    heads: int = 4           # Igual TOPO
    dropout: float = 0.1     # Igual TOPO
    
    # Edge processing
    edge_dim: int = 2        # Igual TOPO
    
    # Activation
    activation: str = 'leaky_relu'
    negative_slope: float = 0.2


@dataclass
class TrainingConfig:
    """Configuração de treinamento."""
    
    # Otimização
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Batch
    chunk_size: int = 100_000
    
    # Curriculum (5 fases)
    num_phases: int = 5
    epochs_per_phase: List[int] = field(default_factory=lambda: [20, 25, 25, 20, 15])
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Spatial CV
    num_folds: int = 5
    buffer_km: float = 2.0


@dataclass
class PathConfig:
    """Caminhos do projeto."""
    
    # Base
    base_dir: Path = Path("f:/arpia_topo_refinado/TOPO_RF/GNN_RF_V2")
    
    @property
    def data_dir(self) -> Path:
        return self.base_dir / "01_data"
    
    @property
    def models_dir(self) -> Path:
        return self.base_dir / "02_models"
    
    @property
    def training_dir(self) -> Path:
        return self.base_dir / "03_training"
    
    @property
    def baselines_dir(self) -> Path:
        return self.base_dir / "04_baselines"
    
    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "05_outputs"
    
    @property
    def evaluation_dir(self) -> Path:
        return self.base_dir / "06_evaluation"
    
    @property
    def paper_dir(self) -> Path:
        return self.base_dir / "07_paper"
    
    @property
    def checkpoints_dir(self) -> Path:
        return self.base_dir / "checkpoints"
    
    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"
    
    # Datasets externos
    gnn_topo_checkpoint: Path = Path("f:/arpia_topo_refinado/v21/checkpoints")
    gnn_topo_dataset: Path = Path("f:/GRAPH_V18.3")
    anatel_database: Path = Path("f:/arpia_topo_refinado/TOPO_RF/GRAFO")


@dataclass
class Config:
    """Configuração master do projeto."""
    
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    rf_output: RFOutputConfig = field(default_factory=RFOutputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def validate(self) -> bool:
        """Valida configuração contra estrutura real."""
        errors = []
        
        # Verificar dimensões
        if self.terrain.input_dim != 17:
            errors.append(f"terrain.input_dim deve ser 17, got {self.terrain.input_dim}")
        
        if self.terrain.output_dim != 14:
            errors.append(f"terrain.output_dim deve ser 14, got {self.terrain.output_dim}")
        
        if self.terrain.edge_dim != 2:
            errors.append(f"terrain.edge_dim deve ser 2, got {self.terrain.edge_dim}")
        
        if self.terrain.canopy_index != 9:
            errors.append(f"terrain.canopy_index deve ser 9, got {self.terrain.canopy_index}")
        
        if self.model.hidden_dim != 512:
            errors.append(f"model.hidden_dim deve ser 512, got {self.model.hidden_dim}")
        
        if errors:
            for e in errors:
                print(f"❌ Config Error: {e}")
            return False
        
        print("✅ Config validation PASSED")
        return True


# Instância global
config = Config()


if __name__ == "__main__":
    # Validar ao importar
    config.validate()
    
    print(f"\n📊 Configuração GNN-RF Propagation V2")
    print(f"   Terrain features: {config.terrain.input_dim}")
    print(f"   Terrain targets: {config.terrain.output_dim}")
    print(f"   Canopy index: {config.terrain.canopy_index}")
    print(f"   Antenna features: {config.antenna.input_dim}")
    print(f"   RF outputs: {config.rf_output.output_dim}")
    print(f"   Hidden dim: {config.model.hidden_dim}")
    print(f"   Device: {config.device}")
