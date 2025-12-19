import numpy as np
from scipy.signal import hilbert


class AnalogToAnalog:
    """
    Optimized Analog-to-Analog Modulation / Demodulation
    (Behavior identical to baseline)
    """

    def __init__(self, carrier_freq=50, sampling_rate=1000):
        self.fc = float(carrier_freq)
        self.fs = float(sampling_rate)

        # Precompute constants
        self._two_pi_fc = 2.0 * np.pi * self.fc

    # -------------------------------------------------
    # Internal helpers (optimized & cached)
    # -------------------------------------------------

    def _time_axis(self, n):
        return np.arange(n, dtype=np.float64) / self.fs

    def _carrier(self, t):
        return np.cos(self._two_pi_fc * t)

    # -------------------------------------------------
    # MODULATION
    # -------------------------------------------------

    def modulate_am(self, analog_signal, mod_index=1.0):
        x = np.asarray(analog_signal, dtype=np.float64)
        t = self._time_axis(len(x))
        carrier = self._carrier(t)

        # Fast & safe normalization
        peak = np.max(np.abs(x))
        if peak > 0.0:
            x = x / peak

        return t, (1.0 + mod_index * x) * carrier

    def modulate_fm(self, analog_signal, kf=20.0):
        x = np.asarray(analog_signal, dtype=np.float64)
        t = self._time_axis(len(x))

        # Integral (phase accumulator)
        phase_dev = np.cumsum(x) * (2.0 * np.pi * kf / self.fs)
        phase = self._two_pi_fc * t + phase_dev

        return t, np.cos(phase)

    def modulate_pm(self, analog_signal, kp=2.0):
        x = np.asarray(analog_signal, dtype=np.float64)
        t = self._time_axis(len(x))

        phase = self._two_pi_fc * t + kp * x
        return t, np.cos(phase)

    # -------------------------------------------------
    # DEMODULATION
    # -------------------------------------------------

    def demodulate_am(self, modulated_signal):
        s = np.abs(np.asarray(modulated_signal, dtype=np.float64))

        # Low-pass via moving average
        win = max(1, int(self.fs / self.fc) * 2)
        kernel = np.ones(win, dtype=np.float64) / win

        y = np.convolve(s, kernel, mode="same")
        return y - np.mean(y)

    def demodulate_fm(self, modulated_signal):
        s = np.asarray(modulated_signal, dtype=np.float64)

        # Analytic signal → phase
        phase = np.unwrap(np.angle(hilbert(s)))

        # dφ/dt → frequency
        inst_freq = np.diff(phase) * (self.fs / (2.0 * np.pi))

        # remove carrier
        demod = inst_freq - self.fc

        return np.append(demod, demod[-1])

    def demodulate_pm(self, modulated_signal, kp=2.0):
        s = np.asarray(modulated_signal, dtype=np.float64)

        phase = np.unwrap(np.angle(hilbert(s)))
        t = self._time_axis(len(s))

        carrier_phase = self._two_pi_fc * t
        msg = (phase - carrier_phase) / kp

        return msg - np.mean(msg)