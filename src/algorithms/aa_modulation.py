import numpy as np
from scipy.signal import hilbert

class AnalogToAnalog:
    def __init__(self, carrier_freq=50, sampling_rate=1000):
        """
        Architecture: The 'Modulator' Object.
        Args:
            carrier_freq (fc): How fast the carrier wave oscillates normally (Hz).
            sampling_rate (fs): The resolution of our simulation (Points per second).
        """
        self.fc = carrier_freq
        self.fs = sampling_rate

    def _get_time_array(self, num_samples):
        """Helper to generate the time axis (t) based on signal length."""
        duration = num_samples / self.fs
        return np.linspace(0, duration, num_samples, endpoint=False)

    def _get_carrier(self, t):
        """Generates the clean, unmodulated Carrier Wave."""
        return np.cos(2 * np.pi * self.fc * t)

    # ==========================================
    # MODULATION (Encoder)
    # ==========================================

    def modulate_am(self, analog_signal, mod_index=1.0):
        """
        AM: s(t) = [1 + na * m(t)] * cos(2*pi*fc*t)
        """
        t = self._get_time_array(len(analog_signal))
        carrier = self._get_carrier(t)
        
        # Normalize message to avoid overmodulation (clipping at 0)
        max_val = np.max(np.abs(analog_signal))
        norm_message = analog_signal / max_val if max_val > 0 else analog_signal
        
        modulated_signal = (1 + mod_index * norm_message) * carrier
        return t, modulated_signal

    def modulate_fm(self, analog_signal, kf=20.0):
        """
        FM: s(t) = cos(2*pi*fc*t + 2*pi*kf * integral(m(t)))
        """
        t = self._get_time_array(len(analog_signal))
        
        # Integrate message to get Phase from Frequency
        integral_message = np.cumsum(analog_signal) / self.fs
        
        phase = 2 * np.pi * self.fc * t + (2 * np.pi * kf * integral_message)
        modulated_signal = np.cos(phase)
        return t, modulated_signal

    def modulate_pm(self, analog_signal, kp=2.0):
        """
        PM: s(t) = cos(2*pi*fc*t + kp * m(t))
        """
        t = self._get_time_array(len(analog_signal))
        
        # Add message directly to Phase
        phase = 2 * np.pi * self.fc * t + (kp * analog_signal)
        modulated_signal = np.cos(phase)
        return t, modulated_signal

    # ==========================================
    # DEMODULATION (Decoder)
    # ==========================================

    def demodulate_am(self, modulated_signal):
        """
        AM Demod: Envelope Detection + Low Pass Filter
        """
        # 1. Rectification
        rectified = np.abs(modulated_signal)
        
        # 2. Simple Moving Average Filter (Low Pass)
        # Window size covers roughly 2 periods of the carrier
        window_size = max(1, int(self.fs / self.fc) * 2) 
        demodulated = np.convolve(rectified, np.ones(window_size)/window_size, mode='same')
        
        # Remove DC offset (approximate)
        return demodulated - np.mean(demodulated)

    def demodulate_fm(self, modulated_signal):
        """
        FM Demod: Instantaneous Frequency Extraction
        """
        # 1. Extract Analytic Signal to get Phase
        analytic_signal = hilbert(modulated_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # 2. Differentiate Phase to get Frequency (dPhase/dt)
        # diff returns size N-1, so we calculate freq for intervals
        instantaneous_freq = np.diff(instantaneous_phase) * self.fs / (2.0 * np.pi)
        
        # 3. Subtract Carrier Freq to get Message
        demodulated = instantaneous_freq - self.fc
        
        # Pad with one zero to match original length
        return np.append(demodulated, 0)

    def demodulate_pm(self, modulated_signal, kp=2.0):
        """
        PM Demod: Synchronous Phase Extraction
        """
        # 1. Extract Total Phase
        analytic_signal = hilbert(modulated_signal)
        total_phase = np.unwrap(np.angle(analytic_signal))
        
        # 2. Generate Reference Phase (The "Perfect Clock")
        t = self._get_time_array(len(modulated_signal))
        carrier_phase = 2 * np.pi * self.fc * t
        
        # 3. Subtract Reference to find the Shift (Message)
        # Formula: Message = (Total_Phase - Carrier_Phase) / kp
        demodulated = (total_phase - carrier_phase) / kp
        
        # Center the signal (Remove constant phase drift if any)
        return demodulated - np.mean(demodulated)


# ==========================================
# TEST BLOCK (Run this to verify)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("--- Starting Analog Modulation Test ---")
    
    # 1. Setup
    aa = AnalogToAnalog(carrier_freq=50, sampling_rate=1000)
    
    # 2. Input Signal (A slow 2Hz sine wave)
    duration = 1.0
    t_axis = np.linspace(0, duration, int(duration * 1000), endpoint=False)
    original_msg = np.sin(2 * np.pi * 2 * t_axis)
    
    print(f"Input Signal Length: {len(original_msg)} samples")

    # 3. Run All Mod/Demod Cycles
    
    # --- AM ---
    _, am_mod = aa.modulate_am(original_msg, mod_index=0.8)
    am_demod = aa.demodulate_am(am_mod)
    
    # --- FM ---
    _, fm_mod = aa.modulate_fm(original_msg, kf=30.0)
    fm_demod = aa.demodulate_fm(fm_mod)
    
    # --- PM ---
    _, pm_mod = aa.modulate_pm(original_msg, kp=3.0)
    pm_demod = aa.demodulate_pm(pm_mod, kp=3.0) # Pass kp to scale correctly

    # 4. Visualization
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("Analog Modulation & Demodulation Test", fontsize=16)

    # Plot AM
    axs[0, 0].plot(t_axis, am_mod, 'b')
    axs[0, 0].set_title("AM Modulated Signal")
    axs[0, 1].plot(t_axis, original_msg, 'g--', label="Original")
    axs[0, 1].plot(t_axis, am_demod, 'r', alpha=0.7, label="Demodulated")
    axs[0, 1].set_title("AM Demodulation Result")
    axs[0, 1].legend()

    # Plot FM
    axs[1, 0].plot(t_axis, fm_mod, 'b')
    axs[1, 0].set_title("FM Modulated Signal")
    axs[1, 1].plot(t_axis, original_msg, 'g--', label="Original")
    axs[1, 1].plot(t_axis, fm_demod, 'r', alpha=0.7, label="Demodulated")
    axs[1, 1].set_title("FM Demodulation Result")
    axs[1, 1].legend()

    # Plot PM
    axs[2, 0].plot(t_axis, pm_mod, 'b')
    axs[2, 0].set_title("PM Modulated Signal")
    axs[2, 1].plot(t_axis, original_msg, 'g--', label="Original")
    axs[2, 1].plot(t_axis, pm_demod, 'r', alpha=0.7, label="Demodulated")
    axs[2, 1].set_title("PM Demodulation Result")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()
    
    print("--- Test Completed. Check the popup graph. ---")