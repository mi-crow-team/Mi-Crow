from dataclasses import dataclass


@dataclass
class NeuronText:
    score: float
    text: str
    token_idx: int
    token_str: str