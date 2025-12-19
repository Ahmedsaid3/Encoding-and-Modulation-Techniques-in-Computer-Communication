import numpy as np

class AnalogToAnalog:
    def __init__(self, carrier_freq=100, sampling_rate=1000):
        """
        Architecture: The 'Modulator' Object.
        It holds the configuration for the Carrier Wave (The "Runner" or "Vehicle").
        
        Args:
            carrier_freq (fc): How fast the carrier wave oscillates normally (Hz).
            sampling_rate (fs): The resolution of our simulation (Points per second).
        """
        self.fc = carrier_freq
        self.fs = sampling_rate

    def _get_time_array(self, duration):
        """Helper to generate the time axis (t) based on signal length."""
        # Equivalent to the "Sampling Theorem" requirement in the book
        return np.linspace(0, duration, int(duration * self.fs), endpoint=False)

    def _get_carrier(self, t):
        """Generates the clean, unmodulated Carrier Wave."""
        # s(t) = cos(2 * pi * fc * t)
        return np.cos(2 * np.pi * self.fc * t)

    def modulate_am(self, analog_signal, mod_index=1.0):
        """
        Amplitude Modulation (AM)
        Theory: The 'Envelope' Pattern.
        Formula: s(t) = [1 + na * m(t)] * cos(2*pi*fc*t)
        
        Logic: 
        We simply multiply the carrier by the message. 
        If message is High -> Carrier gets Tall.
        If message is Low  -> Carrier gets Short.
        """
        duration = len(analog_signal) / self.fs
        t = self._get_time_array(duration)
        carrier = self._get_carrier(t)
        
        # 1. Normalize message to ensure it fits in -1 to 1 range (Safety check)
        # This prevents "Overmodulation" (where the signal crosses zero and inverts)
        max_val = np.max(np.abs(analog_signal))
        if max_val > 0:
            norm_message = analog_signal / max_val
        else:
            norm_message = analog_signal

        # 2. Apply Modulation
        # The (1 + ...) part creates the "Offset" so the wave grows/shrinks 
        # around the base size, rather than disappearing at 0.
        modulated_signal = (1 + mod_index * norm_message) * carrier
        
        return t, modulated_signal


    def modulate_pm(self, analog_signal, kp=2.0):
        """
        Phase Modulation (PM)
        Theory: The 'Teleport' Pattern (Time Shift).
        Formula: s(t) = Ac * cos(2*pi*fc*t + kp * m(t))
        
        Logic:
        We do NOT integrate. We add the message directly to the phase angle.
        Message value HIGH -> Teleport Forward (Advance Phase).
        Message value LOW  -> Teleport Backward (Delay Phase).
        """
        duration = len(analog_signal) / self.fs
        t = self._get_time_array(duration)
        
        # 1. Direct Phase Manipulation
        # No cumsum here. The message directly dictates the offset.
        phase = 2 * np.pi * self.fc * t + (kp * analog_signal)
        modulated_signal = np.cos(phase)
        
        return t, modulated_signal

    # --- Simple Demodulation Implementations (For completing the cycle) ---

    def demodulate_am(self, modulated_signal):
        """
        AM Demodulation (Envelope Detector)
        Logic: Take absolute value to flip negative parts up, then 'smooth' it out.
        """
        # 1. Rectification (Absolute Value)
        rectified = np.abs(modulated_signal)
        
        # 2. Low Pass Filter (Smoothing) - Simulating a capacitor
        # A simple moving average filter
        window_size = int(self.fs / self.fc) * 2 # Smooth over a few carrier cycles
        demodulated = np.convolve(rectified, np.ones(window_size)/window_size, mode='same')
        
        # Remove DC offset (The +1 we added earlier) roughly
        return demodulated - np.mean(demodulated)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Setup - Create the Architect Object
    aa = AnalogToAnalog(carrier_freq=50, sampling_rate=1000)

    # 2. Create Input Analog Data (The Message)
    # Let's make a 2Hz Sine Wave (slow message)
    duration = 1.0
    t_msg = np.linspace(0, duration, int(duration * 1000), endpoint=False)
    message_signal = np.sin(2 * np.pi * 2 * t_msg) 

    # 3. Perform Modulations
    # AM: Look for the "Envelope" (Shape matches message)
    _, am_sig = aa.modulate_am(message_signal, mod_index=0.8)
    
    # PM: Look for the "Shift" (Harder to see on sine, but similar to FM)
    _, pm_sig = aa.modulate_pm(message_signal, kp=3.0)

    # 4. Plotting
    plt.figure(figsize=(10, 8))

    # Plot Message
    plt.subplot(4, 1, 1)
    plt.plot(t_msg, message_signal, 'g', lw=2)
    plt.title("Input: Analog Data (Message Signal)")
    plt.grid(True, alpha=0.3)

    # Plot AM
    plt.subplot(4, 1, 2)
    plt.plot(t_msg, am_sig, 'b')
    plt.title("AM Output (Amplitude Changes)")
    plt.ylabel("Voltage")
    plt.grid(True, alpha=0.3)

    # Plot PM
    plt.subplot(4, 1, 3)
    plt.plot(t_msg, pm_sig, 'm')
    plt.title("PM Output (Phase/Position Shifts)")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

import numpy as np
