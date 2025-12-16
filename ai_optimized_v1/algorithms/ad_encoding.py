import numpy as np

class AnalogToDigital:
    def __init__(self):
        pass

    # -----------------------------
    # PCM
    # -----------------------------
    def encode_pcm(self, analog_signal, n_bits=3, min_val=-1, max_val=1):
        x = np.asarray(analog_signal, dtype=float)
        n_bits = int(n_bits)
        L = 2 ** n_bits

        normalized = (x - min_val) / (max_val - min_val)
        idx = np.round(normalized * (L - 1)).astype(int)
        idx = np.clip(idx, 0, L - 1)

        # orijinal format(index, f"0{n_bits}b") ile aynı bit sırası
        shifts = np.arange(n_bits - 1, -1, -1, dtype=int)
        bits_mat = ((idx[:, None] >> shifts[None, :]) & 1).astype(int)
        encoded_bits = bits_mat.reshape(-1)

        return np.asarray(encoded_bits, dtype=int), np.asarray(idx, dtype=int)

    def decode_pcm(self, encoded_bits, n_bits=3, min_val=-1, max_val=1):
        n_bits = int(n_bits)
        b = np.asarray(encoded_bits, dtype=int).reshape(-1)
        L = 2 ** n_bits

        m = (b.size // n_bits) * n_bits
        if m == 0:
            return np.array([], dtype=float)

        bits_reshaped = b[:m].reshape(-1, n_bits)
        powers = 2 ** np.arange(n_bits)[::-1]
        idx = (bits_reshaped * powers).sum(axis=1)

        decoded = (idx / (L - 1)) * (max_val - min_val) + min_val
        return np.asarray(decoded, dtype=float)

    # -----------------------------
    # Delta Modulation (stateful ama prealloc ile hızlandı)
    # -----------------------------
    def encode_delta_modulation(self, analog_signal, delta=0.1, initial_value=0.0):
        x = np.asarray(analog_signal, dtype=float)
        delta = float(delta)

        bits = np.empty(x.size, dtype=int)
        reconstructed = np.empty(x.size, dtype=float)

        x_hat = float(initial_value)
        for i, sample in enumerate(x):
            if sample >= x_hat:
                bits[i] = 1
                x_hat += delta
            else:
                bits[i] = 0
                x_hat -= delta
            reconstructed[i] = x_hat

        return bits, reconstructed

    def decode_delta_modulation(self, bits, delta=0.1, initial_value=0.0):
        b = np.asarray(bits, dtype=int).reshape(-1)
        delta = float(delta)

        steps = np.where(b == 1, delta, -delta).astype(float)
        reconstructed = np.cumsum(steps) + float(initial_value)
        return np.asarray(reconstructed, dtype=float)
