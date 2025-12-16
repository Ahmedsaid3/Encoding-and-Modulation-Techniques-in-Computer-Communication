import numpy as np

class AnalogToDigital:
    def __init__(self):
        pass

    def encode_pcm(self, analog_signal, n_bits=3, min_val=-1, max_val=1):
        """
        AI OPTIMIZATION:
        - Removed string formatting loop (Very slow).
        - Used Bitwise operations to extract bits from integers directly.
        """
        L = 2 ** n_bits
        
        # Vectorized Quantization
        normalized = (analog_signal - min_val) / (max_val - min_val)
        q_indices = np.round(normalized * (L - 1)).astype(int)
        q_indices = np.clip(q_indices, 0, L - 1)
        
        # Vectorized Integer to Bit Array
        # e.g. 6 (110) -> [1, 1, 0]
        # We broadcast bitwise shifts
        # Shifts: [2, 1, 0] for 3 bits
        shifts = np.arange(n_bits - 1, -1, -1)
        
        # (N, 1) >> (1, n_bits) -> (N, n_bits)
        # Then & 1 to isolate the bit
        bit_matrix = (q_indices[:, None] >> shifts[None, :]) & 1
        
        return bit_matrix.flatten(), q_indices

    def decode_pcm(self, encoded_bits, n_bits=3, min_val=-1, max_val=1):
        """
        AI OPTIMIZATION:
        - Reshape and dot product for bits-to-int conversion.
        """
        L = 2 ** n_bits
        bits_reshaped = np.array(encoded_bits).reshape(-1, n_bits)
        
        # Vectorized Binary to Decimal
        # [1, 0, 1] dot [4, 2, 1] -> 5
        powers = 2 ** np.arange(n_bits)[::-1]
        q_indices = np.dot(bits_reshaped, powers)
        
        decoded_voltages = (q_indices / (L - 1)) * (max_val - min_val) + min_val
        return decoded_voltages

    def encode_delta_modulation(self, analog_signal, delta=0.1, initial_value=0.0):
        """
        NOTE: DM Encoding is inherently sequential (feedback loop).
        x_hat[n] depends on x_hat[n-1].
        Pure vectorization is impossible without JIT (like Numba).
        We keep the loop but optimize array access.
        """
        # Using vanilla loop as feedback prevents simple numpy vectorization
        x = np.array(analog_signal)
        n = len(x)
        bits = np.zeros(n, dtype=int)
        reconstructed = np.zeros(n, dtype=float)
        
        x_hat = initial_value
        
        # Optimized loop (Standard Python is slow here, but necessary w/o Numba)
        for i in range(n):
            if x[i] >= x_hat:
                bits[i] = 1
                x_hat += delta
            else:
                bits[i] = 0
                x_hat -= delta
            reconstructed[i] = x_hat
            
        return bits, reconstructed

    def decode_delta_modulation(self, bits, delta=0.1, initial_value=0.0):
        """
        AI OPTIMIZATION:
        - Decoding is just summing up the steps!
        - Replaced loop with np.cumsum (Cumulative Sum).
        """
        bits = np.array(bits)
        
        # Map bits: 1 -> +delta, 0 -> -delta
        steps = np.where(bits == 1, delta, -delta)
        
        # Cumulative sum gives the signal path
        reconstructed = np.cumsum(steps) + initial_value
        
        return reconstructed