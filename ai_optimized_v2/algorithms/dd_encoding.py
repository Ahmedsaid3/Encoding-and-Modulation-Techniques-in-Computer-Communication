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
    

    def encode_nrzi(self, bits, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION NOTE:
        NRZI depends on transition. 
        Instead of a loop, we can use 'cumsum' on bits modulo 2 to track state changes.
        But since voltage inverts (+ -> - -> +), we need cumulative mapping.
        Formula: (-1) ^ cumsum(bits)
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # Cumsum gives the total number of 1s encountered so far.
        # If total 1s is Even -> Polarity is same as start.
        # If total 1s is Odd -> Polarity is inverted.
        # Let's assume start is Positive.
        # (-1)**cumsum -> +1 if even, -1 if odd. This works perfectly!
        
        states = np.power(-1, np.cumsum(bits)) * self.amplitude 
        
        # But wait, NRZI transition happens for '1'. 
        # If bit is 0, state stays same. If bit is 1, state flips.
        # The cumsum logic above correctly models this flip-flop behavior.
        
        # We need to prepend the initial state if we want strict transition logic, 
        # but typically we assume the first bit causes transition relative to an implicit previous state.
        # The logic above: 
        # Bits: [1, 0, 1] -> Cumsum: [1, 1, 2] -> Power: [-1, -1, +1]
        # 1 -> Flip to neg (Correct)
        # 0 -> Stay neg (Correct)
        # 1 -> Flip to pos (Correct)
        
        signal = np.repeat(states, points_per_bit)
        total_points = len(signal)
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points, endpoint=False)
        
        return time_axis, signal

    def decode_nrzi(self, signal, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION:
        Compare current level with previous level using array shifting.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        n_bits = len(signal) // points_per_bit
        valid_signal = signal[:n_bits * points_per_bit]
        reshaped = valid_signal.reshape(-1, points_per_bit)
        
        # Take sample from middle
        current_levels = reshaped[:, points_per_bit // 2]
        
        # We need previous levels to compare.
        # Shift current_levels to right by 1 to get "previous"
        # [L0, L1, L2] -> [Start, L0, L1]
        previous_levels = np.roll(current_levels, 1)
        previous_levels[0] = self.amplitude # Assumption from encoder start
        
        # Transition Logic:
        # If (prev > 0 and curr < 0) OR (prev < 0 and curr > 0) -> Bit 1
        # Simplified: If prev * curr < 0 (signs are different) -> Bit 1
        
        decoded_bits = np.where(previous_levels * current_levels < 0, 1, 0)
        return decoded_bits

    # ============================================
    # 5. PSEUDOTERNARY (Optimized) - NEW!
    # ============================================
    def encode_pseudoternary(self, bits, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION:
        Similar to AMI but for Zeros. 
        Bit 1 -> 0V
        Bit 0 -> Alternating
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        time_axis = np.linspace(0, len(bits) * bit_duration, len(bits) * points_per_bit, endpoint=False)
        
        # Logic is AMI inverted: track Zeros
        zeros_mask = (bits == 0)
        polarity_tracker = np.cumsum(zeros_mask)
        
        bit_levels = np.zeros_like(bits, dtype=float)
        
        # Start with -self.amplitude as "last zero voltage" implies first 0 becomes Positive?
        # Friend's code: last_zero = -Amp. First 0 -> current = -(-Amp) = +Amp.
        # So: Odd counts -> +Amp, Even counts -> -Amp
        
        odd_zeros = (polarity_tracker % 2 == 1) & zeros_mask
        even_zeros = (polarity_tracker % 2 == 0) & zeros_mask
        
        bit_levels[odd_zeros] = self.amplitude
        bit_levels[even_zeros] = -self.amplitude
        # Bit 1s are already 0 in bit_levels init
        
        signal = np.repeat(bit_levels, points_per_bit)
        return time_axis, signal

    def decode_pseudoternary(self, signal, baud_rate=1, sampling_rate=100):
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        n_bits = len(signal) // points_per_bit
        valid_signal = signal[:n_bits * points_per_bit]
        reshaped = valid_signal.reshape(-1, points_per_bit)
        
        mid_values = reshaped[:, points_per_bit // 2]
        
        # If abs < 0.5 -> 1, Else -> 0
        decoded_bits = np.where(np.abs(mid_values) < 0.5, 1, 0)
        return decoded_bits

    # ============================================
    # 6. DIFFERENTIAL MANCHESTER (Optimized) - NEW!
    # ============================================
    def encode_dif_manch(self, bits, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION:
        Complex state machine. 
        Bit 1: Same first half as previous second half (No Transition at start)
        Bit 0: Inverted first half (Transition at start)
        ALWAYS Transition in middle.
        
        Vectorizing this dependency (state[i] depends on state[i-1]) is hard without loop.
        However, we can mathematically model the "Start Level" of each bit.
        
        Start Level Flips if:
        - Previous bit ended High -> Next bit starts High? No.
        - Logic:
          - If Bit=0 -> Transition at boundary -> Polarity Flips relative to prev end.
          - If Bit=1 -> No Transition at boundary -> Polarity Same as prev end.
        
        Actually, let's look at the pattern of "Left Halves":
        L[i] depends on R[i-1].
        R[i] is always -L[i] (Middle transition).
        
        So we just need L[i].
        If Bit[i] == 1 -> L[i] = R[i-1] = -L[i-1] (Invert prev left)
        If Bit[i] == 0 -> L[i] = -R[i-1] = L[i-1] (Same as prev left)
        
        Wait, let's re-read friend's code:
        if bit==1: append [amp]*left (same amp) -> amp stays same.
        else: amp = -amp; append [amp]*left (flip amp)
        after half: amp = -amp (middle transition)
        
        Friend's code logic trace:
        'amplitude' variable tracks the level.
        For bit 1: uses current 'amplitude' for left.
        For bit 0: flips 'amplitude' then uses for left.
        Then ALWAYS flips 'amplitude' for right.
        
        This means 'amplitude' variable represents the level at the END of the previous bit?
        No, it represents "Current active level".
        
        Let's use Numba or just simple logical vectorization.
        Actually, simple cumulative logic works:
        
        Transition at start happens if bit == 0.
        Transition at middle happens ALWAYS.
        
        Let's track the "Left Half Polarity".
        L[i] relation to L[i-1]:
        Friend's code implies:
        Start of loop: 'amplitude' is the level of R[i-1].
        Bit 1: L[i] = R[i-1]. (No transition at boundary)
        Bit 0: L[i] = -R[i-1]. (Transition at boundary)
        Since R[i-1] = -L[i-1] (always mid transition),
        
        Bit 1: L[i] = -L[i-1]
        Bit 0: L[i] = -(-L[i-1]) = L[i-1]
        
        Conclusion:
        - If Bit is 1 -> Left Polarity Inverts.
        - If Bit is 0 -> Left Polarity Stays.
        
        This is exactly NRZI logic!
        We can use cumsum on (bits==1) to determine Left Polarities.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        half_points = points_per_bit // 2
        
        # 1. Determine Left Half Polarities
        # Logic derived: L[i] flips if bit is 1. L[i] stays if bit is 0.
        # This is (-1)^cumsum(bits == 1)
        # Initial amplitude is self.amplitude.
        
        # cumsum starts from index 0. 
        # But for first bit: 
        # Friend code: if bit=1, signal.extend([amp]). if 0, amp=-amp..
        # This means initial logic is slightly different or assumes virtual prev bit is 1?
        # Let's match Friend's "state change" logic exactly.
        # Friend Logic:
        # 0 -> Change state
        # 1 -> Keep state
        # This is inverted NRZI logic.
        
        # Let's map "Change Actions":
        # Bit 0 -> Flip
        # Bit 1 -> No Flip
        # Map bits to 0/1 for "Flip Action": (1 - bits) gives 1 for 0, 0 for 1.
        
        flip_actions = 1 - bits
        # Cumsum the flips
        flip_counts = np.cumsum(flip_actions)
        
        # Calculate Left Levels: initial_amp * (-1)^(flip_counts)
        # Friend starts loop with 'amplitude'.
        # If first bit is 1 (flip=0) -> amp stays -> L[0] = +Amp
        # If first bit is 0 (flip=1) -> amp flips -> L[0] = -Amp
        # This matches (-1)^flip_counts
        
        left_levels = self.amplitude * np.power(-1, flip_counts)
        right_levels = -left_levels # Always transition in middle
        
        # Combine Left and Right
        # Stack them: [[L0, R0], [L1, R1]...]
        levels = np.stack((left_levels, right_levels), axis=1)
        flat_levels = levels.flatten()
        
        signal = np.repeat(flat_levels, half_points)
        
        # Fix length
        expected_len = len(bits) * points_per_bit
        if len(signal) < expected_len:
            signal = np.pad(signal, (0, expected_len - len(signal)), 'edge')
        elif len(signal) > expected_len:
            signal = signal[:expected_len]
            
        time_axis = np.linspace(0, len(bits) * bit_duration, len(signal), endpoint=False)
        return time_axis, signal

    def decode_dif_manch(self, signal, baud_rate=1, sampling_rate=100):
        """
        AI OPTIMIZATION:
        Compare Prev Right with Cur Left.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        half_points = points_per_bit // 2
        
        n_bits = len(signal) // points_per_bit
        valid_signal = signal[:n_bits * points_per_bit]
        reshaped = valid_signal.reshape(-1, points_per_bit)
        
        left_sample_idx = half_points // 2
        right_sample_idx = half_points + (half_points // 2)
        
        cur_lefts = reshaped[:, left_sample_idx]
        cur_rights = reshaped[:, right_sample_idx]
        
        # Need prev_rights
        prev_rights = np.roll(cur_rights, 1)
        prev_rights[0] = self.amplitude # Initial state assumption
        
        # Friend Logic:
        # if (prev_right > 0 and cur_left > 0) or (prev_right < 0 and cur_left < 0) -> 1
        # i.e., if signs are SAME -> 1
        # else -> 0
        
        decoded_bits = np.where(prev_rights * cur_lefts > 0, 1, 0)
        return decoded_bits