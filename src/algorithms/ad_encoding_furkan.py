import numpy as np

class AnalogToDigital:
    def __init__(self):
        pass

    def encode_delta_modulation(self, analog_signal, delta=0.1, initial_value=0.0):
        """
        Delta Modulation (DM) Encoding
        Rule: if x[n] >= x_hat[n-1] -> output 1 and step up by +delta
              else                 -> output 0 and step down by -delta
        Returns:
            bits: 0/1 stream
            reconstructed: running staircase approximation at encoder
        """
        x = np.array(analog_signal, dtype=float)

        bits = []
        x_hat = initial_value
        reconstructed = []

        for sample in x:
            if sample >= x_hat:
                bits.append(1)
                x_hat += delta
            else:
                bits.append(0)
                x_hat -= delta
            reconstructed.append(x_hat)

        return np.array(bits, dtype=int), np.array(reconstructed, dtype=float)

    def decode_delta_modulation(self, bits, delta=0.1, initial_value=0.0):
        """
        Delta Modulation (DM) Decoding
        Rule: bit 1 -> +delta, bit 0 -> -delta
        Returns:
            reconstructed: staircase signal
        """
        b = np.array(bits, dtype=int)

        x_hat = initial_value
        reconstructed = []

        for bit in b:
            x_hat += delta if bit == 1 else -delta
            reconstructed.append(x_hat)

        return np.array(reconstructed, dtype=float)


# --- TEST BLOCK ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ad = AnalogToDigital()

    # Make a sampled analog signal
    duration = 1.0
    fs = 50
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    analog_signal = 0.9 * np.sin(2 * np.pi * 2 * t)  # 2 Hz sine

    delta = 0.08
    bits, enc_recon = ad.encode_delta_modulation(analog_signal, delta=delta, initial_value=0.0)
    dec_recon = ad.decode_delta_modulation(bits, delta=delta, initial_value=0.0)

    print("First 30 DM bits:", bits[:30].tolist())

    plt.figure(figsize=(10, 5))
    plt.plot(t, analog_signal, label="Original Analog", linewidth=2, alpha=0.5)
    plt.step(t, dec_recon, where="mid", label=f"Delta Mod Reconstruct (Δ={delta})", linewidth=2)
    plt.title("Delta Modulation (DM)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
