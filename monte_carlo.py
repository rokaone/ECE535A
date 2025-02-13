import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import factorial as f

from numpy._core.numerictypes import float64

def ideal_rep_code(n, p, a):
    p_e = 0
    for k in range(0,int((n-1)/2) + 1):
        p_e = float(f(n)/(f(k)*f(n-k))) * np.pow(1-a,n-k) * np.pow(a,k)
    return p_e

def repetition_code(symbol: int, n: int) -> list:
    return [symbol] * n

def bpsk(symbol, symbol_energy) -> list:
    return [symbol * symbol_energy]

class Generator:
    def __init__(self, symbols: list, p: list) -> None:
        self.symbols = symbols
        self.p = p
        print(f"Generator - syms: {symbols} p_syms: {p}")

    def gen(self) -> int:
        return int(np.random.choice(self.symbols, p=self.p))

class Channel:
    def __init__(self, mean: float, noise_power: float, alpha: float) -> None:
        self.mean = mean
        self.noise_power = noise_power
        self.alpha = [alpha, 1-alpha]
        print(f"Channel - mean: {mean} noise_pwr: {noise_power} alpha: {alpha}")

    def apply_channel(self, symbol: int) -> float:
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

    def bpsk(self, mean: float, received: float, expected: int) -> bool:
        r_hat = int(received > mean)
        return r_hat == expected

class MonteCarlo:
    def __init__(self, confidence: float, target_p_e: float, rel_error: float, repetitions: int = None, alpha: float = 0.0) -> None:
        self.confidence = confidence
        self.target_p_e = target_p_e
        self.rel_error = rel_error
        self.repetitions = repetitions
        self.alpha = alpha
        print(f"Sim: confidence: {confidence} target_p_e: {target_p_e} rel_error: {rel_error} repetitions: {repetitions}")

    def calc_n_trials(self) -> int:
        z_alpha = norm.ppf(1 - (1 - self.confidence) / 2)
        N = (z_alpha**2 * (1 - self.target_p_e)) / (self.rel_error**2 * self.target_p_e)
        return np.ceil(N)

    def run_trials(self) -> float:
        N = int(self.calc_n_trials())
        g = Generator([0,1], [0.5, 0.5])
        chan = Channel(0, 0.1, self.alpha)
        dec = Decoder()
        bit_err = 0
        print(f"Running {N} Trials")
        for i in range(N):
            bit = g.gen()
            bit_enc = repetition_code(bit, self.repetitions)
            bit_chan = chan.flip_bit(bit_enc)
            bit_dec = dec.repetition_code(bit_chan, bit_enc)
            if bit_dec != bit:
                #print(f"ERROR - Trial {i}: {bit=} {bit_enc=} {bit_chan=} {bit_dec=} {bit_err=}")
                bit_err += 1

        print(f"p_e = {bit_err / N}")
        return bit_err / N



if __name__=="__main__":

    for i in range(1, 9, 2):
        p_error = []
        p_error_diff = []
        alphas = []
        for alpha in range(5, 55, 5):
            mc = MonteCarlo(0.9, 1e-2, 0.1, i, alpha/100)
            p_e = mc.run_trials()
            p_error.append(p_e)
            p_error_diff.append(np.abs(p_e - ideal_rep_code(i, 0.5, alpha/100)))
            alphas.append(alpha/100)
        print('-'*11)
        plt.figure(1)
        plt.loglog(alphas, p_error, label=f"n: {i}", marker='x')
        plt.figure(2)
        plt.plot(alphas, p_error_diff, label=f"n: {i}", marker='x')
    
    plt.legend()
    plt.show()
    print("Done")
