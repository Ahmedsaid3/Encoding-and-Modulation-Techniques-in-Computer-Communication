import numpy as np

class DigitalToDigital:
    def __init__(self):
        self.amplitude = 5

    def encode_nrz_l(self, bits, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION NOTE:
        - Removed 'for' loops (Vectorization).
        - Used 'np.repeat' to expand bits to signal duration instantly.
        - Used 'np.where' for condition checking on the whole array at once.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # OPTIMIZATION 1: Pre-calculation of Time Axis
        total_points = len(bits) * points_per_bit
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points, endpoint=False)
        
        # OPTIMIZATION 2: Vectorized Expansion (No Loop)
        # Repeat each bit 'points_per_bit' times. 
        # Ex: bits=[0, 1], points=2 -> extended_bits=[0, 0, 1, 1]
        extended_bits = np.repeat(bits, points_per_bit)
        
        # OPTIMIZATION 3: Masking instead of If-Else
        # Where bit is 0 -> +Amplitude, Where bit is 1 -> -Amplitude
        signal = np.where(extended_bits == 0, self.amplitude, -self.amplitude)
        
        return time_axis, signal

    def decode_nrz_l(self, signal, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION NOTE:
        - Used reshaping and mean calculation instead of iterating chunks.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # Ensure signal length is divisible by points_per_bit (truncate excess)
        n_bits = len(signal) // points_per_bit
        valid_signal = signal[:n_bits * points_per_bit]
        
        # OPTIMIZATION: Reshape 1D signal to 2D Matrix (Rows=Bits, Cols=Samples)
        # This allows us to process all bits in parallel.
        reshaped_signal = valid_signal.reshape(-1, points_per_bit)
        
        # Calculate mean of each row (bit duration)
        # Axis=1 means "calculate average across columns for each row"
        row_means = reshaped_signal.mean(axis=1)
        
        # Vectorized Decision
        # If mean > 0 -> 0, Else -> 1
        decoded_bits = np.where(row_means > 0, 0, 1)
        
        return decoded_bits

    def encode_bipolar_ami(self, bits, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION NOTE:
        - This is tricky because AMI has 'memory' (previous 1 state).
        - We use 'cumsum' (cumulative sum) on masked bits to simulate memory without a loop.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        total_points = len(bits) * points_per_bit
        
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points, endpoint=False)
        
        # 1. Identify locations of '1's
        ones_mask = (bits == 1)
        
        # 2. Assign alternating polarity using Cumulative Sum
        # cumsum on ones_mask: [0, 1, 0, 1, 1] -> [0, 1, 1, 2, 3]
        # We want odd numbers to be +V, even numbers to be -V (or vice versa)
        polarity_tracker = np.cumsum(ones_mask)
        
        # Calculate levels for bits:
        # If bit is 0 -> 0V
        # If bit is 1:
        #    Check if its index in the 'ones' sequence is odd or even
        #    (polarity_tracker % 2) gives 0 or 1. Map this to +1/-1.
        #    Formula: -1 if even, +1 if odd (or customizable)
        
        # Create an array of levels for each bit
        bit_levels = np.zeros_like(bits, dtype=float)
        
        # Only update where bits are 1
        # Logic: If cumsum is odd -> +1, if even -> -1. 
        # (1 - 2 * (track % 2)) converts {1->-1, 0->1}.. let's simplify:
        # let's say we want odd counts to be positive.
        odd_ones = (polarity_tracker % 2 == 1) & ones_mask
        even_ones = (polarity_tracker % 2 == 0) & ones_mask
        
        bit_levels[odd_ones] = self.amplitude
        bit_levels[even_ones] = -self.amplitude
        
        # 3. Expand to full signal
        signal = np.repeat(bit_levels, points_per_bit)
        
        return time_axis, signal

    def decode_bipolar_ami(self, signal, baud_rate=1, sampling_rate=100):
        # Same logic as NRZ decode: Reshape and check absolute value
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        n_bits = len(signal) // points_per_bit
        valid_signal = signal[:n_bits * points_per_bit]
        
        reshaped = valid_signal.reshape(-1, points_per_bit)
        
        # Take the value from the middle of each bit period
        mid_indices = points_per_bit // 2
        mid_values = reshaped[:, mid_indices]
        
        # If absolute value is close to 0 -> 0, else -> 1
        decoded_bits = np.where(np.abs(mid_values) < 0.5, 0, 1)
        
        return decoded_bits

    def encode_manchester(self, bits, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION NOTE:
        - Used Kronecker Product (np.kron) or repetition trick.
        - Manchester pattern is fixed: 
          0 -> [1, -1] sequence
          1 -> [-1, 1] sequence
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        half_points = points_per_bit // 2
        
        # Create a pattern matrix for each bit
        # We need two halves for each bit. 
        # Logic: 
        # Bit 0 -> [+V, -V]
        # Bit 1 -> [-V, +V]
        
        # Let's create an array of shape (N_bits, 2) representing the two halves
        # Initialize with Amplitude
        levels = np.zeros((len(bits), 2))
        
        # Fill for Bit 0 (High -> Low)
        levels[bits == 0, 0] = self.amplitude
        levels[bits == 0, 1] = -self.amplitude
        
        # Fill for Bit 1 (Low -> High)
        levels[bits == 1, 0] = -self.amplitude
        levels[bits == 1, 1] = self.amplitude
        
        # Now we have the levels, we need to expand them to full duration.
        # Each "half" in 'levels' needs to be repeated 'half_points' times.
        
        # Flatten the levels to get the sequence of transitions
        flat_levels = levels.flatten() # [Bit1_Half1, Bit1_Half2, Bit2_Half1...]
        
        # Repeat each half-level
        # Note: If points_per_bit is odd, this is slightly inexact but fine for optimization demo.
        # To be precise, we repeat elements.
        signal = np.repeat(flat_levels, half_points)
        
        # Handling odd points_per_bit (fix length mismatch)
        expected_len = len(bits) * points_per_bit
        if len(signal) < expected_len:
            signal = np.pad(signal, (0, expected_len - len(signal)), 'edge')
        elif len(signal) > expected_len:
            signal = signal[:expected_len]

        time_axis = np.linspace(0, len(bits) * bit_duration, len(signal), endpoint=False)
        return time_axis, signal
        
    def decode_manchester(self, signal, baud_rate=1, sampling_rate=100):
        # Optimized Decoding using Slice Comparison
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        half_points = points_per_bit // 2
        
        n_bits = len(signal) // points_per_bit
        valid_signal = signal[:n_bits * points_per_bit]
        reshaped = valid_signal.reshape(-1, points_per_bit)
        
        # Extract sample points from 1st and 2nd half using array indexing
        first_samples = reshaped[:, half_points // 2]
        second_samples = reshaped[:, half_points + (half_points // 2)]
        
        # Vectorized Logic:
        # 0 if (First > 0 and Second < 0)
        # 1 if (First < 0 and Second > 0)
        decoded_bits = np.where((first_samples > 0) & (second_samples < 0), 0, 1)
        
        return decoded_bits