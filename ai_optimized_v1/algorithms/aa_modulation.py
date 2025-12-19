import numpy as np


class AnalogToAnalog:
    def __init__(self, carrier_freq=100, sampling_rate=1000):
        """
        Carrier config:
          fc: carrier frequency (Hz)
          fs: sampling rate (samples/sec)
        """
        self.fc = float(carrier_freq)
        self.fs = float(sampling_rate)

        # speed: precompute angular carrier frequency
        self._wc = 2.0 * np.pi * self.fc

        # speed: cache AM smoothing kernel by window size
        self._kernel_cache = {}

    def _get_time_array(self, duration):
        """Generate time axis exactly like original: linspace(..., endpoint=False)."""
        n = int(duration * self.fs)
        return np.linspace(0.0, duration, n, endpoint=False)

    def _get_carrier(self, t):
        """Clean carrier: cos(2*pi*fc*t)."""
        return np.cos(self._wc * t)

    def modulate_am(self, analog_signal, mod_index=1.0):
        """
        AM: s(t) = [1 + mod_index * m_norm(t)] * cos(2*pi*fc*t)
        (Normalization logic preserved.)
        """
        x = np.asarray(analog_signal, dtype=float)
        duration = x.size / self.fs

        t = self._get_time_array(duration)
        carrier = self._get_carrier(t)

        max_val = np.max(np.abs(x))
        if max_val > 0:
            norm_message = x / max_val
        else:
            norm_message = x

        modulated_signal = (1.0 + (float(mod_index) * norm_message)) * carrier
        return t, modulated_signal

    def modulate_pm(self, analog_signal, kp=2.0):
        """
        PM: s(t) = cos(2*pi*fc*t + kp*m(t))
        """
        x = np.asarray(analog_signal, dtype=float)
        duration = x.size / self.fs

        t = self._get_time_array(duration)
        phase = (self._wc * t) + (float(kp) * x)
        modulated_signal = np.cos(phase)
        return t, modulated_signal

    def demodulate_am(self, modulated_signal):
        """
        AM demod (envelope detector):
          1) abs()
          2) moving-average LPF via np.convolve (same as original behavior)
          3) remove DC offset
        """
        s = np.asarray(modulated_signal, dtype=float)

        rectified = np.abs(s)

        window_size = int(self.fs / self.fc) * 2
        if window_size < 1:
            window_size = 1

        kernel = self._kernel_cache.get(window_size)
        if kernel is None:
            kernel = np.ones(window_size, dtype=float) / float(window_size)
            self._kernel_cache[window_size] = kernel

        demodulated = np.convolve(rectified, kernel, mode="same")
        return demodulated - np.mean(demodulated)


# ----------------------------
# (Optional) Demo / Visualization
# ----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)

    message = np.zeros_like(t)
    message[250:500] = 1
    message[750:1000] = -1

    aa = AnalogToAnalog(carrier_freq=10, sampling_rate=fs)

    # AM demo
    tt, am_sig = aa.modulate_am(message, mod_index=1.0)
    am_demod = aa.demodulate_am(am_sig)

    # PM demo (teleport)
    tt, pm_sig = aa.modulate_pm(message, kp=2.0 * np.pi * 0.5)

    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, message, lw=2)
    plt.title("Message")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(tt, am_sig)
    plt.title("AM Signal")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(tt, pm_sig)
    plt.title("PM Signal")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()