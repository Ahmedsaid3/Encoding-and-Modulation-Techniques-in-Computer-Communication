import numpy as np

class DigitalToDigital:
    def __init__(self):
        self.amplitude = 5  # aynı

    # -----------------------------
    # NRZ-L
    # -----------------------------
    def encode_nrz_l(self, bits, baud_rate=1, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        levels = np.where(bits == 0, self.amplitude, -self.amplitude).astype(float, copy=False)
        signal = np.repeat(levels, ppb)

        # ORİJİNALİN endpoint=True davranışını koruyoruz (default)
        time_axis = np.linspace(0, len(bits) * bit_duration, signal.size)
        return time_axis, signal

    def decode_nrz_l(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        mid = chunks[:, ppb // 2]
        return np.where(mid > 0, 0, 1).astype(int)

    # -----------------------------
    # Bipolar AMI
    # -----------------------------
    def encode_bipolar_ami(self, bits, baud_rate=1, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        ones = (bits == 1).astype(np.int32)
        count = np.cumsum(ones)                       # 1'lerin sayısı
        # 1'lerde +,-,+,- ... (ilk 1 pozitif) => last_one_voltage başlangıç -A idi
        sign = np.where(bits == 1, np.where((count % 2) == 1, 1.0, -1.0), 0.0)
        levels = sign * float(self.amplitude)

        signal = np.repeat(levels, ppb)
        time_axis = np.linspace(0, len(bits) * bit_duration, signal.size)
        return time_axis, signal

    def decode_bipolar_ami(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        mid = chunks[:, ppb // 2]
        return (np.abs(mid) >= 0.5).astype(int)

    # -----------------------------
    # Manchester
    # -----------------------------
    def encode_manchester(self, bits, baud_rate=1, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)
        half = ppb // 2

        # bit 0: + then -, bit 1: - then +
        first = np.where(bits == 0, self.amplitude, -self.amplitude).astype(float, copy=False)
        second = -first

        out = np.empty(bits.size * ppb, dtype=float)
        out2d = out.reshape(bits.size, ppb)
        out2d[:, :half] = first[:, None]
        out2d[:, half:] = second[:, None]

        time_axis = np.linspace(0, len(bits) * bit_duration, out.size)
        return time_axis, out

    def decode_manchester(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)
        half = ppb // 2

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        first_sample = chunks[:, half // 2]
        second_sample = chunks[:, half + (half // 2)]

        out = np.zeros(n, dtype=int)
        out[(first_sample < 0) & (second_sample > 0)] = 1
        # diğer durumlar orijinalde 0’a düşüyordu
        return out

    # -----------------------------
    # NRZI
    # -----------------------------
    def encode_nrzi(self, bits, baud_rate=1, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        flips = (bits == 1).astype(np.int32)
        # current_level başlangıç +A; her 1'de flip
        parity = np.cumsum(flips) % 2
        levels = np.where(parity == 0, self.amplitude, -self.amplitude).astype(float)

        signal = np.repeat(levels, ppb)
        time_axis = np.linspace(0, len(bits) * bit_duration, signal.size)
        return time_axis, signal

    def decode_nrzi(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        level = chunks[:, ppb // 2]

        prev = np.empty(n, dtype=float)
        prev[0] = self.amplitude
        prev[1:] = level[:-1]

        transition = ((prev > 0) & (level < 0)) | ((prev < 0) & (level > 0))
        return transition.astype(int)

    # -----------------------------
    # Pseudoternary
    # -----------------------------
    def encode_pseudoternary(self, bits, baud_rate=1, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        zeros = (bits == 0).astype(np.int32)
        count0 = np.cumsum(zeros)
        # bit 0 => alternating +,-,+,- ...  (ilk 0 pozitif)
        sign0 = np.where(bits == 0, np.where((count0 % 2) == 1, 1.0, -1.0), 0.0)
        levels = sign0 * float(self.amplitude)

        signal = np.repeat(levels, ppb)
        time_axis = np.linspace(0, len(bits) * bit_duration, signal.size)
        return time_axis, signal

    def decode_pseudoternary(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        ppb = int(sampling_rate * bit_duration)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        mid = chunks[:, ppb // 2]
        # 0V -> bit 1, nonzero -> bit 0 (orijinal)
        return np.where(np.abs(mid) < 0.5, 1, 0).astype(int)

    # -----------------------------
    # Differential Manchester (IEEE 802.5)
    # (orijinal davranışı birebir koruyacak şekilde vektörize)
    # -----------------------------
    def encode_dif_manch(self, bits, baud_rate=1, sampling_rate=100):
        bits = np.asarray(bits, dtype=np.uint8)

        bit_duration = 1 / baud_rate
        ppb_f = bit_duration * sampling_rate
        if ppb_f < 4:
            raise ValueError("sampling_rate too low for baud_rate; need ≥ 4 samples per bit")

        left_half = int((ppb_f - 1) // 2)
        right_half = int(ppb_f - left_half)
        ppb = left_half + right_half

        b = bits.astype(np.int8, copy=False)

        # a_i = A * (-1)^(count_ones_before_i)
        ones = (b == 1).astype(np.int32)
        cnt = np.cumsum(ones)
        cnt_before = np.empty_like(cnt)
        cnt_before[0] = 0
        if cnt.size > 1:
            cnt_before[1:] = cnt[:-1]
        a_sign = np.where((cnt_before % 2) == 0, 1.0, -1.0)

        # left_level = a_i if bit==1 else -a_i
        left_level = float(self.amplitude) * a_sign * (2.0 * b - 1.0)
        right_level = -left_level

        out = np.empty(bits.size * ppb, dtype=float)
        out2d = out.reshape(bits.size, ppb)
        out2d[:, :left_half] = left_level[:, None]
        out2d[:, left_half:] = right_level[:, None]

        time_axis = np.linspace(0, len(bits) * bit_duration, out.size)
        return time_axis, out

    def decode_dif_manch(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1 / baud_rate
        ppb = int(bit_duration * sampling_rate)
        if ppb < 4:
            raise ValueError("sampling_rate too low for baud_rate; need ≥ 4 samples per bit")

        left_half = (ppb - 1) // 2
        right_half = ppb - left_half

        left_sample_idx = left_half // 2
        right_sample_idx = left_half + (right_half // 2)

        s = np.asarray(signal, dtype=float)
        n = s.size // ppb
        if n == 0:
            return np.array([], dtype=int)

        chunks = s[: n * ppb].reshape(n, ppb)
        cur_left = chunks[:, left_sample_idx]
        cur_right = chunks[:, right_sample_idx]

        prev_right = np.empty(n, dtype=float)
        prev_right[0] = self.amplitude
        if n > 1:
            prev_right[1:] = cur_right[:-1]

        same_sign = ((prev_right > 0) & (cur_left > 0)) | ((prev_right < 0) & (cur_left < 0))
        return same_sign.astype(int)
