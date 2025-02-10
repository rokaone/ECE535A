import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from dataclasses import dataclass

from numpy._core.numerictypes import float64

@dataclass
class DecoderError:
    num_error: int
    num_total: int

class Generator:
    def __init__(self, symbols: list, p: int) -> None:
        self.symbols = symbols
        self.p = p

    def gen(self) -> int:
        return int(np.random.choice(self.symbols, p=self.p))

class Channel:
    def __init__(self, mean: float64, noise_power: float64, alpha: float64) -> None:
        self.mean = mean
        self.noise_power = noise_power
        self.alpha = alpha

    def apply_channel(self, symbol: int) -> np.float64:
        noise = np.sqrt(self.noise_power) * np.random.randn() + self.mean
        return symbol + noise

    def flip_bit(self, symbol: int) -> int:
        return (symbol * np.random.choice([0, 1], p=self.alpha)) % 2

class Encoder:
    def repetition_code(self, symbol: int, n: int) -> list:
        return [symbol] * n

    def bpsk(self, symbol, symbol_energy) -> list:
        return [symbol * symbol_energy]

class Decoder:
    def repetition_code(self, received: list, expected: list) -> DecoderError:
        num_errors = 0
        for i in range(len(received)):
            if received[i] != expected[i]:
                num_errors += 1
        return DecoderError(num_errors, len(received))

    def bpsk(self, mean: float64, received: float64, expected: int) -> bool:
        r_hat = int(received > mean)
        return r_hat == expected

class MonteCarlo:
    def __init__(self, target_confidence: float, std_dev: float, margin: float) -> None:
        self.target_confidence = target_confidence
        self.std_dev = std_dev
        self.margin = margin

    def calc_n_trials(self) -> int:
        z_a =
        return np.ceil()

    def run_trials(self) -> None:
        print(f"Running Trials")

if __name__=="__main__":

    print("Done")
