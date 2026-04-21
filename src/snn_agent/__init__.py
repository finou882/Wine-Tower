"""
snn_agent package init
"""
from .environment import MultipleTMaze
from .lif import LIFLayer
from .stdp import STDPSynapse
from .wta import WTALayer
from .agent import SNNAgent
from .curriculum import GoalCurriculum
from .trainer import LifelongTrainer

__all__ = [
    "MultipleTMaze",
    "LIFLayer",
    "STDPSynapse",
    "WTALayer",
    "SNNAgent",
    "GoalCurriculum",
    "LifelongTrainer",
]
