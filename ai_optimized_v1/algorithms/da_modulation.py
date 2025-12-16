import numpy as np

class DigitalToAnalog:
    def __init__(self):
        self.amplitude = 1  # aynı

    # -----------------------------
    # ASK
    # -----------------------------
    def modulate_ask(self, bits, baud_rate=1, carrier_freq=5, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        total_duration = len(bits) * bit_duration

        total_points = int(total_duration * sampling_rate)
        time_axis = np.linspace(0, total_duration, total_points, endpoint=False)

        carrier_signal = np.sin(2 * np.pi * carrier_freq * time_axis)

        ppb = int(sampling_rate * bit_duration)
        env = np.repeat(bits.astype(float) * self.amplitude, ppb)

        # orijinal rounding farklarını bozmamak için hizala
        if env.size < total_points:
            env = np.pad(env, (0, total_points - env.size), mode="constant")
        else:
            env = env[:total_points]

        modulated_signal = carrier_signal * env
        return time_axis, modulated_signal

    def demodulate_ask(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        expected_max_energy = np.sum(
            np.abs(np.sin(2 * np.pi * carrier_freq * np.linspace(0, bit_duration, ppb)))
        )
        threshold = expected_max_energy * 0.5

        chunks = s[: n * ppb].reshape(n, ppb)
        energy = np.sum(np.abs(chunks), axis=1)
        return (energy > threshold).astype(int)

    # -----------------------------
    # MFSK
    # (orijinalin "son parça eksikse padding yok" davranışını korur)
    # -----------------------------
    def modulate_mfsk(self, bits, M=4, baud_rate=1, base_freq=5, freq_sep=3, sampling_rate=1000):
        bits = np.asarray(bits, dtype=np.uint8)
        k = int(np.log2(M))
        bit_duration = 1 / baud_rate
        pps = int(bit_duration * sampling_rate)

        full = (len(bits) // k) * k
        syms = []
        if full > 0:
            resh = bits[:full].reshape(-1, k)
            powers = (2 ** np.arange(k)[::-1]).astype(np.uint32)
            syms = (resh.astype(np.uint32) * powers).sum(axis=1).tolist()

        # remainder: orijinal gibi "kadar bit" ile symbol oluştur
        rem = len(bits) - full
        if rem > 0:
            symbol = 0
            for b in bits[full:]:
                symbol = (symbol << 1) | int(b)
            syms.append(symbol)

        if len(syms) == 0:
            return np.array([]), np.array([])

        t = np.linspace(0, bit_duration, pps, endpoint=False)
        freqs = base_freq + np.asarray(syms, dtype=float) * freq_sep
        seg = np.sin(2 * np.pi * freqs[:, None] * t[None, :])

        signal = seg.reshape(-1)
        time_axis = np.linspace(0, len(syms) * bit_duration, signal.size, endpoint=False)
        return time_axis, signal

    def demodulate_mfsk(self, signal, M=4, baud_rate=1, base_freq=5, freq_sep=3, sampling_rate=1000):
        k = int(np.log2(M))
        bit_duration = 1 / baud_rate
        pps = int(bit_duration * sampling_rate)

        s = np.asarray(signal, dtype=float)
        n = s.size // pps
        if n == 0:
            return np.array([], dtype=int)

        t = np.linspace(0, bit_duration, pps, endpoint=False)
        refs = np.stack(
            [np.sin(2 * np.pi * (base_freq + i * freq_sep) * t) for i in range(M)],
            axis=0
        )

        chunks = s[: n * pps].reshape(n, pps)
        scores = chunks @ refs.T
        sym = np.argmax(scores, axis=1)

        decoded_bits = []
        for v in sym.tolist():
            for b in range(k - 1, -1, -1):
                decoded_bits.append((v >> b) & 1)
        return np.asarray(decoded_bits, dtype=int)

    # -----------------------------
    # DPSK
    # -----------------------------
    def modulate_dpsk(self, bits, baud_rate=1, carrier_freq=5, sampling_rate=1000):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1 / baud_rate
        ppb = int(bit_duration * sampling_rate)

        phase = np.cumsum((bits == 1).astype(np.int32)) * np.pi
        t = np.linspace(0, bit_duration, ppb, endpoint=False)
        seg = np.sin(2 * np.pi * carrier_freq * t[None, :] + phase[:, None])

        signal = seg.reshape(-1)
        time_axis = np.linspace(0, len(bits) * bit_duration, signal.size, endpoint=False)
        return time_axis, signal

    def demodulate_dpsk(self, signal, baud_rate=1, carrier_freq=None, sampling_rate=1000):
        bit_duration = 1 / baud_rate
        ppb = int(bit_duration * sampling_rate)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n <= 1:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        corr = np.sum(chunks[:-1] * chunks[1:], axis=1)
        return np.where(corr > 0, 0, 1).astype(int)

    # -----------------------------
    # BFSK
    # -----------------------------
    def modulate_bfsk(self, bits, baud_rate=1, freq_0=5, freq_1=10, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        total_duration = len(bits) * bit_duration
        total_points = int(total_duration * sampling_rate)

        time_axis = np.linspace(0, total_duration, total_points, endpoint=False)

        c0 = np.sin(2 * np.pi * freq_0 * time_axis)
        c1 = np.sin(2 * np.pi * freq_1 * time_axis)

        ppb = int(sampling_rate * bit_duration)
        mask = np.repeat(bits, ppb)
        if mask.size < total_points:
            mask = np.pad(mask, (0, total_points - mask.size), mode="constant")
        else:
            mask = mask[:total_points]

        fsk_signal = np.where(mask == 0, c0, c1) * self.amplitude
        return time_axis, fsk_signal

    def demodulate_bfsk(self, signal, baud_rate=1, freq_0=5, freq_1=10, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        t_ref = np.linspace(0, bit_duration, ppb, endpoint=False)
        ref0 = np.sin(2 * np.pi * freq_0 * t_ref)
        ref1 = np.sin(2 * np.pi * freq_1 * t_ref)

        chunks = s[: n * ppb].reshape(n, ppb)
        score0 = np.sum(chunks * ref0, axis=1)
        score1 = np.sum(chunks * ref1, axis=1)

        return np.where(score1 > score0, 1, 0).astype(int)

    # -----------------------------
    # MPSK
    # -----------------------------
    def modulate_mpsk(self, bits, M=2, baud_rate=1, carrier_freq=5, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        k = int(np.log2(M))

        rem = len(bits) % k
        if rem != 0:
            bits = np.concatenate([bits, np.zeros(k - rem, dtype=np.uint8)])

        resh = bits.reshape(-1, k)
        powers = 2 ** np.arange(k)[::-1]
        symbols = (resh * powers).sum(axis=1)

        sym_dur = 1.0 / baud_rate
        pps = int(sampling_rate * sym_dur)
        t = np.linspace(0, sym_dur, pps, endpoint=False)

        phase = symbols * (2 * np.pi / M)
        seg = self.amplitude * np.sin(2 * np.pi * carrier_freq * t[None, :] + phase[:, None])

        sig = seg.reshape(-1)
        time_axis = np.linspace(0, len(symbols) * sym_dur, sig.size, endpoint=False)
        return time_axis, sig

    def demodulate_mpsk(self, signal, M=2, baud_rate=1, carrier_freq=5, sampling_rate=100):
        k = int(np.log2(M))
        sym_dur = 1.0 / baud_rate
        pps = int(sampling_rate * sym_dur)

        s = np.asarray(signal, dtype=float)
        n = s.size // pps
        if n == 0:
            return np.array([], dtype=int)

        t_ref = np.linspace(0, sym_dur, pps, endpoint=False)
        refs = np.stack(
            [np.sin(2 * np.pi * carrier_freq * t_ref + (m * 2 * np.pi / M)) for m in range(M)],
            axis=0
        )

        chunks = s[: n * pps].reshape(n, pps)
        scores = chunks @ refs.T
        best = np.argmax(scores, axis=1)

        out = []
        for sym in best.tolist():
            bstr = format(int(sym), f"0{k}b")
            out.extend(int(c) for c in bstr)
        return np.asarray(out, dtype=int)
