import numpy as np
from scipy.stats import norm
#import matplotlib.pyplot as plt
from dataclasses import dataclass

from numpy._core.numerictypes import float64

def repetition_code(symbol: int, n: int) -> list:
    return [symbol] * n

def bpsk(symbol, symbol_energy) -> list:
    return [symbol * symbol_energy]

class Generator:
    def __init__(self, symbols: list, p: [int]) -> None:
        self.symbols = symbols
        self.p = p

    def gen(self) -> int:
        print(self.symbols)
        print(self.p)
        return int(np.random.choice(self.symbols, p=self.p))

class Channel:
    def __init__(self, mean: float64, noise_power: float64, alpha: float64) -> None:
        self.mean = mean
        self.noise_power = noise_power
        self.alpha = [alpha, 1-alpha]

    def apply_channel(self, symbol: int) -> np.float64:
        noise = np.sqrt(self.noise_power) * np.random.randn() + self.mean
        return symbol + noise

    def flip_bit(self, symbols: int) -> int:
        sym_out = []
        for sym in symbols:
            sym_out.append(int((sym * np.random.choice([0, 1], p=self.alpha)) % 2))
        return sym_out

class Decoder:
    def repetition_code(self, received: list, expected: list) -> int:
        total = 0
        for bit in received:
            total += bit
        return int(total > np.floor(len(received) / 2))

    def bpsk(self, mean: float64, received: float64, expected: int) -> bool:
        r_hat = int(received > mean)
        return r_hat == expected

class MonteCarlo:
    def __init__(self, confidence: float, std_dev: float, target_p_e: float, rel_error: float) -> None:
        self.confidence = confidence
        self.std_dev = std_dev
        self.target_p_e = target_p_e
        self.rel_error = rel_error

    def calc_n_trials(self) -> int:
        z_alpha = norm.ppf(1 - (1 - self.confidence) / 2)
        N = (z_alpha**2 * (1 - self.target_p_e)) / (self.rel_error**2 * self.target_p_e)
        return np.ceil(N)

    def run_trials(self) -> None:
        N = int(self.calc_n_trials())
        g = Generator([0,1], [0.5, 0.5])
        chan = Channel(0, 0.1, 0.9)
        dec = Decoder()
        bit_err = 0
        print(f"Running {N} Trials")
        for i in range(N):
            bit = g.gen()
            bit_enc = repetition_code(bit, 2)
            bit_chan = chan.flip_bit(bit_enc)
            bit_dec = dec.repetition_code(bit_chan, bit_enc)
            bit_err += int(bit_dec == bit)
            print(f"Trial {i}: {bit=} {bit_enc=} {bit_chan=} {bit_dec=} {bit_err=}")



if __name__=="__main__":
    
    mc = MonteCarlo(0.9, 0, 1e-1, 0.1)
    mc.run_trials()
    print("Done")
