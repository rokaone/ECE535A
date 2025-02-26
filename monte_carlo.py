import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.special import factorial as f
import multiprocessing as mp


def ideal_rep_code(n, a):
    p_e = 0
    for k in range(0, int((n - 1) / 2) + 1):
        p_e += float(f(n) / (f(k) * f(n - k))) * np.pow(1 - a, n - k) * np.pow(a, k)
    return 1 - p_e


def repetition_code(symbol: int, n: int) -> list:
    return [symbol] * n


def bpsk(symbol, symbol_energy) -> list:
    symbols = []
    for sym in symbol:
        symbols.append((1 - 2 * sym) * symbol_energy)
    return symbols


class Generator:
    def __init__(self, symbols: list, p: list) -> None:
        self.symbols = symbols
        self.p = p

    def gen(self) -> int:
        return int(np.random.choice(self.symbols, p=self.p))


class Channel:
    def __init__(self, mean: float, noise_power: float, alpha: float) -> None:
        self.mean = mean
        self.noise_power = noise_power
        self.alpha = [1 - alpha, alpha]

    def apply_channel(self, symbol: list) -> list:
        symbols = []
        for sym in symbol:
            noise = np.sqrt(self.noise_power) * np.random.randn() + self.mean
            symbols.append(sym + noise)
        return symbols

    def flip_bit(self, symbols: list) -> list:
        return [int(np.abs(1 - sym)) if np.random.choice([0, 1], p=self.alpha) else sym for sym in symbols]


class Decoder:
    def repetition_code(self, received: list) -> int:
        rec_dec = []
        for rec in received:
            if rec >= 0:
                rec_dec.append(0)
            else:
                rec_dec.append(1)
        return int(sum(rec_dec) > np.floor(len(rec_dec) / 2))

    def bpsk(self, mean: float, received: float) -> int:
        return int(received < mean)


class MonteCarlo:
    def __init__(self, confidence: float, target_p_e: float, rel_error: float) -> None:
        self.confidence = confidence
        self.target_p_e = target_p_e
        self.rel_error = rel_error

    def calc_n_trials(self) -> int:
        z_alpha = norm.ppf(1 - (1 - self.confidence) / 2)
        N = (z_alpha**2 * (1 - self.target_p_e)) / (self.rel_error**2 * self.target_p_e)
        return int(np.ceil(N))

    def run_single_trial_rep_code(self, n, N):
        """Function to run a single trial with a given n"""
        print("Starting trial n:", n)
        g = Generator([0, 1], [0.5, 0.5])
        dec = Decoder()
        bit_energy = 1
        p_error = []
        p_error_diff = []
        sigmas = []
        p_error_expected = []
        if (n >= 11):
            self.target_p_e = 1e-4
            N = self.calc_n_trials()

        for sigma in range(50, 1000, 50):
            chan = Channel(0, sigma / 100, 0)
            bit_err = 0

            for _ in range(N):
                bit = g.gen()
                bit_enc = repetition_code(bit, n)
                bit_map = bpsk(bit_enc, bit_energy)
                bit_chan = chan.apply_channel(bit_map)
                bit_dec = dec.repetition_code(bit_chan)
                if bit_dec != bit:
                    bit_err += 1

            p_e = bit_err / N
            p_e_expected = ideal_rep_code(n, sigma / 100)
            p_error.append(p_e)
            p_error_expected.append(float(p_e_expected))
            p_error_diff.append(np.abs(p_e - p_e_expected))
            sigmas.append(sigma / 100)

        print("Finished trial n:", n)
        return n, sigmas, p_error, p_error_expected  # Return results for plotting

    def run_trials_rep_code_parallel(self, repetitions: int) -> None:
        """Parallelized version of run_trials_rep_code"""
        N = self.calc_n_trials()
        n_values = list(range(1, repetitions + 2, 2))
        num_workers = min(mp.cpu_count(), len(n_values))

        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(self.run_single_trial_rep_code, [(n, N) for n in n_values])

        plt.figure(1)
        for n, sigmas, p_error, p_error_expected in results:
            print(f"n: {n}")
            print(f"{sigmas=}")
            print(f"{p_error=}")
            print('-'*20)
            plt.loglog(sigmas, p_error, label=f"n: {n}", marker="x")

        plt.legend()
        plt.grid(True)
        plt.title("Probability of Error for Repetition Code")
        plt.xlabel(r"$\sigma$")
        plt.ylabel(r"$P_e$")

        plt.figure(2)
        for n, sigmas, _, p_error_diff in results:
            plt.loglog(sigmas, p_error_diff, label=f"p_e error - n: {n}", marker="x")

        plt.legend()
        plt.grid(True)
        plt.title("Error in Probability of Error for Repetition Code")
        plt.xlabel(r"$\sigma$")
        plt.ylabel(r"$Error(P_e)$")

    def run_single_trial_bpsk(self, sigma, N):
        """Function to run a single BPSK trial"""
        print("Starting trial:", sigma)
        g = Generator([0, 1], [0.5, 0.5])
        dec = Decoder()
        chan = Channel(0, sigma / 100, 0)
        bit_err = 0
        bit_energy = 1
        mean = 0

        for _ in range(N):
            bit = g.gen()
            bit_enc = bpsk(bit, bit_energy)
            bit_chan = chan.apply_channel(bit_enc)
            bit_dec = dec.bpsk(mean, bit_chan)
            if bit_dec != bit:
                bit_err += 1

        p_e = bit_err / N
        print("Finished trial:", sigma)
        return sigma / 100, p_e

    def run_trials_bpsk_parallel(self) -> None:
        """Parallelized version of run_trials_bpsk"""
        N = self.calc_n_trials()
        sigma_values = list(range(10, 105 , 5))
        num_workers = min(mp.cpu_count(), len(sigma_values))

        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(self.run_single_trial_bpsk, [(sigma, N) for sigma in sigma_values])

        sigma_vec, p_error_vec = zip(*results)

        plt.figure(3)
        plt.loglog(sigma_vec, p_error_vec, marker="x")
        plt.grid(True)
        plt.title("Probability of Error for BPSK in AWGN Channel")
        plt.xlabel(r"$\sigma$")
        plt.ylabel(r"$P_e$")


if __name__ == "__main__":
    # Fix for multiprocessing error
    mp.set_start_method("spawn", force=True)

    mc = MonteCarlo(0.9, 1e-2, 0.1)
    mc.run_trials_rep_code_parallel(repetitions=9)
    #mc.run_trials_bpsk_parallel()
    plt.show()
    print("Done")

