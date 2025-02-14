import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import factorial as f


def ideal_rep_code(n, a):
    p_e = 0
    for k in range(0,int((n-1)/2) + 1):
        p_e += float(f(n)/(f(k)*f(n-k))) * np.pow(1-a,n-k) * np.pow(a,k)
    return (1 - p_e)

def repetition_code(symbol: int, n: int) -> list:
    return [symbol] * n

def bpsk(symbol, symbol_energy) -> float:
    return (1 - 2*symbol) * symbol_energy

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
        self.alpha = [1-alpha, alpha]
        #print(f"Channel - mean: {mean} noise_pwr: {noise_power} alpha: {alpha}")

    def apply_channel(self, symbol: float) -> float:
        noise = np.sqrt(self.noise_power) * np.random.randn() + self.mean
        return symbol + noise

    def flip_bit(self, symbols: list) -> list:
        sym_out = []
        for sym in symbols:
            sym_chan = sym
            flip = int(np.random.choice([0,1], p=self.alpha))
            if flip:
                sym_chan = int(np.abs(1 - sym))
            sym_out.append(sym_chan)
            #print(f"{flip=} {sym} -> {sym_chan}")
        return sym_out

class Decoder:
    def repetition_code(self, received: list) -> int:
        total = 0
        for bit in received:
            total += bit
        return int(total > np.floor(len(received) / 2))

    def bpsk(self, mean: float, received: float) -> int:
        return int(received < mean)

class MonteCarlo:
    def __init__(self, confidence: float, target_p_e: float, rel_error: float) -> None:
        self.confidence = confidence
        self.target_p_e = target_p_e
        self.rel_error = rel_error
        print(f"Sim: confidence: {confidence} target_p_e: {target_p_e} rel_error: {rel_error}")

    def calc_n_trials(self) -> int:
        z_alpha = norm.ppf(1 - (1 - self.confidence) / 2)
        N = (z_alpha**2 * (1 - self.target_p_e)) / (self.rel_error**2 * self.target_p_e)
        return int(np.ceil(N))

    def run_trials_rep_code(self, repetitions: int) -> None:
        N = int(self.calc_n_trials())
        g = Generator([0,1], [0.5, 0.5])
        dec = Decoder()
        for n in range(1, repetitions+2, 2):
            print(f"Repetitions: {n}")
            p_error = []
            p_error_diff = []
            alphas = []
            p_error_expected = []
            for alpha in range(5, 55, 5):
                chan = Channel(0, 0.1, alpha/100)
                bit_err = 0
                #print(f"Running {N} Trials")
                for i in range(N):
                    bit = g.gen()
                    bit_enc = repetition_code(bit, n)
                    bit_chan = chan.flip_bit(bit_enc)
                    bit_dec = dec.repetition_code(bit_chan)
                    if bit_dec != bit:
                        #print(f"ERROR   - Trial {i}: {bit=} {bit_enc=} {bit_chan=} {bit_dec=} {bit_err=}")
                        bit_err += 1
                p_e = bit_err / N
                print(f"\talpha: {alpha/100} | P_e: {p_e}")
                p_e_expected = ideal_rep_code(n, alpha/100)
                p_error.append(p_e)
                p_error_expected.append(p_e_expected)
                p_error_diff.append(np.abs(p_e - p_e_expected))
                alphas.append(alpha/100)
            plt.figure(1)
            plt.loglog(alphas, p_error, label=f"n: {n}", marker='x')
            plt.legend()
            plt.figure(2)
            plt.loglog(alphas, p_error_diff, label=f"p_e error - n: {n}", marker='x')
            plt.legend()
    
    def run_trials_bpsk(self) -> None:
        N = int(self.calc_n_trials())
        g = Generator([0,1], [0.5, 0.5])
        dec = Decoder()
        bit_energy = 1
        mean = 0
        
        p_error_vec = []
        sigma_vec = []
        bit_chan_vec = []
        for sigma in range(5, 105, 5):
            chan = Channel(0, sigma/100, 0)
            bit_err = 0
            #print(f"Running {N} Trials")
            for i in range(N):
                bit = g.gen()
                bit_enc = bpsk(bit, bit_energy)
                bit_chan = chan.apply_channel(bit_enc)
                bit_chan_vec.append(bit_chan)
                bit_dec = dec.bpsk(mean, bit_chan)
                if bit_dec != bit:
                    #print(f"ERROR   - Trial {i}: {bit=} {bit_enc=} {bit_chan=} {bit_dec=} {bit_err=}")
                    bit_err += 1
                #else:
                    #print(f"SUCCESS - Trial {i}: {bit=} {bit_enc=} {bit_chan=} {bit_dec=} {bit_err=}")
            p_e = bit_err / N
            print(f"\tsigma: {sigma/100} | P_e: {p_e}")
            p_error_vec.append(p_e)
            sigma_vec.append(sigma/100)
        plt.figure(1)
        plt.loglog(sigma_vec, p_error_vec, marker='x')
        plt.legend()
        plt.figure(2)
        plt.hist(bit_chan_vec, density=True, bins=100)


if __name__=="__main__":

    mc = MonteCarlo(0.9, 1e-3, 0.1)
    #mc.run_trials_rep_code(11)
    mc.run_trials_bpsk()
    plt.show()
    print("Done")
