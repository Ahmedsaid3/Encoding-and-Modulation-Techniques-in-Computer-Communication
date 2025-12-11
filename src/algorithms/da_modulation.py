import numpy as np

class DigitalToAnalog:
    def __init__(self):
        self.amplitude = 1  # Maksimum taşıyıcı genliği (1 Volt)

    def modulate_ask(self, bits, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        Binary ASK (Amplitude Shift Keying) Modulation.
        Bit 1 -> Carrier Signal with Amplitude A (Full Power)
        Bit 0 -> Carrier Signal with Amplitude 0 (No Power) - also known as OOK (On-Off Keying)
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        
        # Total duration of the transmission
        total_duration = len(bits) * bit_duration
        
        # Generate the time axis for the entire signal
        total_points = int(total_duration * sampling_rate)
        time_axis = np.linspace(0, total_duration, total_points, endpoint=False)
        
        # Create the carrier signal (pure sine wave)
        # sin(2 * pi * f * t)
        carrier_signal = np.sin(2 * np.pi * carrier_freq * time_axis)
        
        # Create an envelope signal (amplitude mask) based on bits
        # If bit is 1, envelope is 1. If bit is 0, envelope is 0.
        envelope = np.zeros_like(time_axis)
        points_per_bit = int(sampling_rate * bit_duration)
        
        for i, bit in enumerate(bits):
            start_idx = i * points_per_bit
            end_idx = (i + 1) * points_per_bit
            
            # If bit is 1, amplitude is 1. If 0, amplitude is 0.
            if bit == 1:
                envelope[start_idx:end_idx] = self.amplitude
            else:
                envelope[start_idx:end_idx] = 0  # Or use 0.2 * self.amplitude for lower power
                
        # Modulation: Multiply carrier with the envelope
        modulated_signal = carrier_signal * envelope
        
        return time_axis, modulated_signal

    def demodulate_ask(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        ASK Demodulation (Non-coherent detection).
        Logic:
        1. Rectify the signal (take absolute value) to see energy clearly.
        2. Integrate (sum) the energy over the bit duration.
        3. Threshold: If energy is high -> 1, if low -> 0.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        demodulated_bits = []
        
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i : i + points_per_bit]
            
            if len(chunk) < points_per_bit:
                break
            
            # Energy calculation: Sum of absolute values (or squared values)
            chunk_energy = np.sum(np.abs(chunk))
            
            # Dynamic Thresholding:
            # Maximum possible energy = sum of full sine wave absolute values
            # Let's approximate a threshold. Since 0 bit has 0 energy, anything significant is 1.
            # A safe threshold is usually 50% of the expected max energy.
            
            expected_max_energy = np.sum(np.abs(np.sin(2 * np.pi * carrier_freq * np.linspace(0, bit_duration, points_per_bit))))
            threshold = expected_max_energy * 0.5
            
            if chunk_energy > threshold:
                demodulated_bits.append(1)
            else:
                demodulated_bits.append(0)
                
        return np.array(demodulated_bits)
    

# test the implementation
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    da = DigitalToAnalog()
    
    # Test Data
    original_bits = [1, 0, 1, 1, 0, 1, 0]
    print(f"Original Bits: {original_bits}")
    
    # --- TEST ASK ---
    # Carrier frequency should be higher than baud rate for good visualization
    # Baud Rate: 1 bit/sec, Carrier Freq: 5 Hz (5 cycles per bit)
    t, s = da.modulate_ask(original_bits, baud_rate=1, carrier_freq=3, sampling_rate=1000)
    
    recovered_bits = da.demodulate_ask(s, baud_rate=1, carrier_freq=3, sampling_rate=1000)
    print(f"Recovered Bits: {list(recovered_bits)}")
    
    if list(original_bits) == list(recovered_bits):
        print("RESULT: SUCCESS! ASK works correctly.")
    else:
        print("RESULT: FAILURE! Data mismatch.")
        
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(t, s)
    plt.title("Amplitude Shift Keying (ASK)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.grid(True)
    
    # Draw bit boundaries
    for x in range(len(original_bits) + 1):
        plt.axvline(x, color='red', linestyle='--', alpha=0.3)
        
    plt.show()