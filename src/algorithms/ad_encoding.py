import numpy as np

class AnalogToDigital:
    def __init__(self):
        pass

    def encode_pcm(self, analog_signal, n_bits=3, min_val=-1, max_val=1):
        """
        Pulse Code Modulation (PCM) Encoding.
        Steps: Sampling -> Quantization -> Encoding
        
        Args:
            analog_signal: The input array (must be sampled properly before passing here)
            n_bits: Bit depth (determines number of levels L = 2^n)
            min_val, max_val: The voltage range of the signal
        """
        # 1. Quantization Setup
        # Calculate number of levels (L)
        L = 2 ** n_bits
        
        # Calculate step size (delta) between levels
        step_size = (max_val - min_val) / L
        
        # Normalize signal to fit in 0 to L-1 range
        # Formula: (signal - min) / (max - min) * (L - 1)
        # We clip properly to avoid out of bounds
        normalized_signal = (analog_signal - min_val) / (max_val - min_val)
        
        # Round to nearest integer (Quantization process)
        quantized_indices = np.round(normalized_signal * (L - 1)).astype(int)
        
        # Ensure indices are within bounds [0, L-1]
        quantized_indices = np.clip(quantized_indices, 0, L - 1)
        
        # 2. Encoding (Decimal to Binary)
        encoded_bits = []
        
        # Format string for binary conversion (e.g., if n_bits=3, format is '03b')
        bin_format = f'0{n_bits}b'
        
        for index in quantized_indices:
            # Convert integer index to binary string
            binary_string = format(index, bin_format)
            # Split string into individual bits and add to list
            encoded_bits.extend([int(b) for b in binary_string])
            
        return np.array(encoded_bits), quantized_indices

    def decode_pcm(self, encoded_bits, n_bits=3, min_val=-1, max_val=1):
        """
        PCM Decoding.
        Steps: Binary -> Decimal Indices -> Voltage Levels (Reconstruction)
        """
        L = 2 ** n_bits
        step_size = (max_val - min_val) / (L - 1) # Using L-1 to map exactly to max_val
        
        # 1. Group bits back into symbols
        # Reshape the 1D bit array into rows of 'n_bits'
        # Example: [1,0,1, 0,1,1] -> [[1,0,1], [0,1,1]] for 3-bit PCM
        bits_reshaped = np.array(encoded_bits).reshape(-1, n_bits)
        
        # 2. Binary to Decimal
        # Powers of two: [4, 2, 1] for 3 bits
        powers_of_two = 2 ** np.arange(n_bits)[::-1]
        quantized_indices = (bits_reshaped * powers_of_two).sum(axis=1)
        
        # 3. Reconstruction (Indices to Voltage)
        # Map integer levels back to voltage range
        decoded_voltages = (quantized_indices / (L - 1)) * (max_val - min_val) + min_val
        
        return decoded_voltages

# --- TEST BLOCK ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    ad = AnalogToDigital()
    
    # 1. Generate a smooth analog signal (Simulated Real World)
    duration = 1.0 # seconds
    sampling_freq = 20 # Hz (Low enough to see the steps clearly)
    t = np.linspace(0, duration, int(duration * sampling_freq))
    
    # A sine wave with amplitude 1
    analog_signal = np.sin(2 * np.pi * 1 * t)
    
    print(f"Original Signal Samples: {len(analog_signal)}")
    
    # 2. PCM Encoding (Analog -> Digital)
    # Using 3-bit resolution (8 levels)
    n_bits = 3
    digital_bits, q_indices = ad.encode_pcm(analog_signal, n_bits=n_bits, min_val=-1, max_val=1)
    
    print(f"Encoded Bits (First 20): {digital_bits[:20]}")
    print(f"Quantized Indices: {q_indices[:10]}...")
    
    # 3. PCM Decoding (Digital -> Analog)
    reconstructed_signal = ad.decode_pcm(digital_bits, n_bits=n_bits, min_val=-1, max_val=1)
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot Original Analog Signal
    plt.plot(t, analog_signal, label='Original Analog', color='blue', alpha=0.5, linewidth=2)
    
    # Plot Reconstructed (Quantized) Signal
    # 'step' plot shows the discrete levels nicely
    plt.step(t, reconstructed_signal, where='mid', label=f'PCM Output ({n_bits}-bit)', color='red', linewidth=2)
    
    plt.title(f"PCM (Pulse Code Modulation) - {n_bits} Bit Resolution")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.legend()
    plt.grid(True)
    
    # Draw horizontal grid lines for quantization levels
    L = 2 ** n_bits
    levels = np.linspace(-1, 1, L)
    for level in levels:
        plt.axhline(level, color='gray', linestyle=':', alpha=0.3)
        
    plt.show()