import numpy as np


class DigitalToDigitalExtended:
    """
    Extended Digital-to-Digital encoding schemes.
    Contains: NRZI, Pseudoternary, Differential Manchester
    """
    
    def __init__(self):
        self.amplitude = 5  # Voltage level (e.g., 5 Volts)

    # ============================================
    # NRZI (Non-Return to Zero Inverted)
    # ============================================

    def encode_nrzi(self, bits, baud_rate=1, sampling_rate=100):
        """
        NRZI Encoding (Non-Return to Zero, Invert on ones):
        - Bit 1: Transition occurs (voltage level inverts)
        - Bit 0: No transition (voltage level stays the same)
        
        This is a differential encoding scheme - data is encoded in transitions,
        not absolute voltage levels. Used in USB.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        signal = []
        current_level = self.amplitude  # Start with positive voltage
        
        for bit in bits:
            if bit == 1:
                current_level = -current_level  # Invert on 1
            # For bit 0: no change
            
            signal_segment = np.ones(points_per_bit) * current_level
            signal.extend(signal_segment)
            
        total_points = len(signal)
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points)
        
        return time_axis, np.array(signal)

    def decode_nrzi(self, signal, baud_rate=1, sampling_rate=100):
        """
        NRZI Decoding:
        - Transition occurred -> Bit 1
        - No transition -> Bit 0
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        decoded_bits = []
        previous_level = self.amplitude  # Assume starting from positive
        
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i : i + points_per_bit]
            
            if len(chunk) == 0:
                break
            
            current_level = chunk[len(chunk) // 2]
            
            # Check if transition occurred
            if (previous_level > 0 and current_level < 0) or (previous_level < 0 and current_level > 0):
                decoded_bits.append(1)
            else:
                decoded_bits.append(0)
            
            previous_level = current_level
            
        return np.array(decoded_bits)

    # ============================================
    # PSEUDOTERNARY
    # ============================================

    def encode_pseudoternary(self, bits, baud_rate=1, sampling_rate=100):
        """
        Pseudoternary Encoding (opposite of Bipolar-AMI):
        - Bit 1: 0V (no signal)
        - Bit 0: Alternating +V and -V
        
        Used in ISDN systems.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        signal = []
        last_zero_voltage = -self.amplitude  # First '0' will be positive
        
        for bit in bits:
            if bit == 1:
                signal_segment = np.zeros(points_per_bit)
            else:
                current_voltage = -last_zero_voltage
                signal_segment = np.ones(points_per_bit) * current_voltage
                last_zero_voltage = current_voltage
            
            signal.extend(signal_segment)
            
        total_points = len(signal)
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points)
        
        return time_axis, np.array(signal)

    def decode_pseudoternary(self, signal, baud_rate=1, sampling_rate=100):
        """
        Pseudoternary Decoding:
        - 0V -> Bit 1
        - Non-zero (+V or -V) -> Bit 0
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        decoded_bits = []
        
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i : i + points_per_bit]
            
            if len(chunk) == 0:
                break
            
            mid_point_value = chunk[len(chunk) // 2]
            
            if abs(mid_point_value) < 0.5:
                decoded_bits.append(1)
            else:
                decoded_bits.append(0)
                
        return np.array(decoded_bits)

    # ============================================
    # DIFFERENTIAL MANCHESTER (IEEE 802.5)
    # ============================================

    def encode_dif_manch(self, bits, baud_rate = 1, sampling_rate = 100) :
        bits = np.array(bits)
        # determine single bit duration -> determine number of points for a bit
        bit_duration = 1 / baud_rate
        points_per_bit = bit_duration * sampling_rate

        if points_per_bit < 4:
            raise ValueError("sampling_rate too low for baud_rate; need ≥ 4 samples per bit")
    
        # create signal buffer and determine left and right
        signal = []
        left_half = (int)((points_per_bit-1) // 2) # enforcing left <= right becasue right includes the middle (transition point) 0 low 1 high 2 high (2 will be the sampling for right half)
        right_half = (int)(points_per_bit - left_half)

        amplitude = self.amplitude
        # self amplitute will determine the initial signal state
        for bit in bits :
            # for each bit if 0 transition at the beginning 
            if(bit == 1):
                # append new left segment to signal
                signal.extend([amplitude] * left_half)
            else:
                # invert then append new left segment to signal
                amplitude = -amplitude
                signal.extend([amplitude] * left_half)
            # after half switch voltage at the next half and append (transtion)  
            amplitude = -amplitude
            signal.extend([amplitude] * right_half)

        total_points = len(signal)
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points)
        
        return time_axis, np.array(signal)


    def decode_dif_manch(self, signal, baud_rate = 1, sampling_rate = 100):
        # determine num points by determining bit duration
        bit_duration = 1 / baud_rate
        points_per_bit = (int)(bit_duration * sampling_rate)

        if points_per_bit < 4:
            raise ValueError("sampling_rate too low for baud_rate; need ≥ 4 samples per bit")
    
        #init decoded bits array
        decoded_bits = []

        prev_right = self.amplitude

        # Calculate half sizes (matching encoder)
        left_half = (points_per_bit - 1) // 2   # e.g., 49 for ppb=100
        right_half = points_per_bit - left_half  # e.g., 51 for ppb=100
    
        # Sample in the MIDDLE of each half, not at the edges
        left_sample_idx = left_half // 2                     # e.g., 49//2 = 24
        right_sample_idx = left_half + (right_half // 2)     # e.g., 49 + 25 = 74

        # for each segment sample mid - 1 and mid + 1  
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i: i + points_per_bit]
            cur_left = chunk[left_sample_idx]
            cur_right = chunk[right_sample_idx]
            # when prev mid + 1 == cur mid - 1 then 1
            # else 0
            # append bit to decoded bits
            if((prev_right > 0 and cur_left > 0) or (prev_right < 0 and cur_left < 0)):
                decoded_bits.append(1)
            else:
                decoded_bits.append(0)
            #update cur to prev
            prev_right = cur_right

        return np.array(decoded_bits)




# ============================================
# TEST SECTION
# ============================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    dd = DigitalToDigitalExtended()
    
    def test_encoding(name, encode_func, decode_func, test_bits):
        """Helper function to test an encoding scheme"""
        print(f"\n{'='*50}")
        print(f"Testing {name}...")
        print(f"Original Bits: {test_bits}")
        
        t, s = encode_func(test_bits)
        recovered_bits = decode_func(s)
        print(f"Recovered Bits: {list(recovered_bits)}")
        
        if list(test_bits) == list(recovered_bits):
            print(f"RESULT: SUCCESS! {name} works correctly.")
        else:
            print(f"RESULT: FAILURE! Data mismatch.")
        
        # Visualization
        plt.figure(figsize=(12, 4))
        plt.step(t, s, where='post', linewidth=2)
        plt.title(f"{name} Encoding")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True)
        plt.ylim(-6, 6)
        plt.axhline(0, color='black', linewidth=0.5)
        for x in range(len(test_bits) + 1):
            plt.axvline(x, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    # Test data
    test_bits = [0, 1, 0, 0, 1, 1, 0, 1]
    
    # Run tests for the 3 extended schemes
    print("="*50)
    print("DIGITAL-TO-DIGITAL EXTENDED ENCODING TESTS")
    print("="*50)
    
    test_encoding("NRZI", dd.encode_nrzi, dd.decode_nrzi, test_bits)
    test_encoding("Pseudoternary", dd.encode_pseudoternary, dd.decode_pseudoternary, test_bits)
    test_encoding("Differential Manchester (IEEE 802.5)", dd.encode_dif_manch, dd.decode_dif_manch, test_bits)
    
    print("\n" + "="*50)
    print("ALL EXTENDED TESTS COMPLETED!")
    print("="*50)