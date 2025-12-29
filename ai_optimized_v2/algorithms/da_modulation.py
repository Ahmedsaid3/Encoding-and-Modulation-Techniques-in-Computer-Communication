import numpy as np

class DigitalToAnalog:
    def __init__(self):
        self.amplitude = 1

    def modulate_ask(self, bits, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        AI OPTIMIZATION:
        - Used np.repeat to create the envelope instantly.
        - Calculated carrier wave once for the whole duration.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # 1. Create Envelope (Mask) without loop
        # bits: [1, 0] -> envelope: [1, 1, ..., 0, 0, ...]
        envelope = np.repeat(bits, points_per_bit) * self.amplitude
        
        # 2. Generate Carrier
        total_points = len(envelope)
        t = np.arange(total_points) / sampling_rate
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        return t, carrier * envelope

    def demodulate_ask(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        AI OPTIMIZATION:
        - Reshape signal into (N_bits, Points_per_bit) matrix.
        - Sum along axis=1 (Energy calculation) in one go.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # Truncate & Reshape
        n_bits = len(signal) // points_per_bit
        signal_matrix = signal[:n_bits * points_per_bit].reshape(n_bits, points_per_bit)
        
        # Energy per bit (Vectorized sum)
        energies = np.sum(np.abs(signal_matrix), axis=1)
        
        # Threshold calculation (Analytical)
        # Average energy of a half-sine rectified is related to 2/pi or just take 50% max
        # Or calculate explicitly for one bit duration
        t_ref = np.arange(points_per_bit) / sampling_rate
        max_energy = np.sum(np.abs(np.sin(2 * np.pi * carrier_freq * t_ref)))
        threshold = max_energy * 0.5
        
        return np.where(energies > threshold, 1, 0)

    def modulate_bfsk(self, bits, baud_rate=1, freq_0=5, freq_1=10, sampling_rate=100):
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # Expand bits to time axis
        extended_bits = np.repeat(bits, points_per_bit)
        total_points = len(extended_bits)
        t = np.arange(total_points) / sampling_rate
        
        # Vectorized Frequency Selection
        # If bit=0 -> freq=freq_0, If bit=1 -> freq=freq_1
        freqs = np.where(extended_bits == 0, freq_0, freq_1)
        
        # Phase correction is tricky in pure vectorization for continuous phase.
        # But for standard FSK (discontinuous is okay or we assume coherent time):
        signal = self.amplitude * np.sin(2 * np.pi * freqs * t)
        
        return t, signal

    def demodulate_bfsk(self, signal, baud_rate=1, freq_0=5, freq_1=10, sampling_rate=100):
        """
        AI OPTIMIZATION:
        - Matrix Multiplication for Correlation.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        n_bits = len(signal) // points_per_bit
        
        # Reshape signal: Rows=Bits, Cols=Samples
        matrix = signal[:n_bits * points_per_bit].reshape(n_bits, points_per_bit)
        
        # Generate Reference Vectors
        t_ref = np.arange(points_per_bit) / sampling_rate
        ref0 = np.sin(2 * np.pi * freq_0 * t_ref)
        ref1 = np.sin(2 * np.pi * freq_1 * t_ref)
        
        # Vectorized Correlation (Dot Product via broadcasting or matmul)
        # sum(chunk * ref) across axis 1
        score0 = np.dot(matrix, ref0)
        score1 = np.dot(matrix, ref1)
        
        return np.where(score1 > score0, 1, 0)

    def modulate_mpsk(self, bits, M=2, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        AI OPTIMIZATION:
        - Replaced symbol-loop with phase array.
        - Calculated sine wave in one go with phase shifts.
        """
        bits = np.array(bits)
        k = int(np.log2(M))
        
        # Pad and Reshape bits to symbols
        remainder = len(bits) % k
        if remainder != 0:
            bits = np.concatenate([bits, np.zeros(k - remainder, dtype=int)])
            
        reshaped_bits = bits.reshape(-1, k)
        powers = 2 ** np.arange(k)[::-1]
        symbols = (reshaped_bits * powers).sum(axis=1)
        
        # Symbols to Phases
        phases = symbols * (2 * np.pi / M)
        
        # Create Time Axis
        symbol_duration = 1.0 / baud_rate
        points_per_symbol = int(sampling_rate * symbol_duration)
        
        # Expand phases to full signal length
        # [P1, P2] -> [P1, P1... P2, P2...]
        extended_phases = np.repeat(phases, points_per_symbol)
        
        total_points = len(extended_phases)
        t = np.arange(total_points) / sampling_rate
        # Note: t needs to reset for each symbol for correct phase visual? 
        # Actually standard carrier is continuous t. But usually in diagrams phase is relative to symbol start.
        # Let's use local t for phase shift clarity (like original code)
        t_local = np.tile(np.arange(points_per_symbol) / sampling_rate, len(symbols))
        
        signal = self.amplitude * np.sin(2 * np.pi * carrier_freq * t_local + extended_phases)
        
        # Return continuous t for x-axis
        t_global = np.arange(total_points) / sampling_rate
        return t_global, signal

    def demodulate_mpsk(self, signal, M=2, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        AI OPTIMIZATION:
        - Massive speedup using Matrix Multiplication against Reference Bank.
        """
        k = int(np.log2(M))
        bit_duration = 1.0 / baud_rate # Wait, MPSK uses symbol rate
        symbol_duration = 1.0 / baud_rate
        points_per_symbol = int(sampling_rate * symbol_duration)
        
        n_symbols = len(signal) // points_per_symbol
        matrix = signal[:n_symbols * points_per_symbol].reshape(n_symbols, points_per_symbol)
        
        # 1. Create Reference Bank Matrix (Rows=M, Cols=Samples)
        t_ref = np.arange(points_per_symbol) / sampling_rate
        # Shape: (M, points)
        phase_shifts = np.arange(M) * (2 * np.pi / M)
        # Using broadcasting to create all references at once
        # (M, 1) + (points,) -> (M, points)
        refs = np.sin(2 * np.pi * carrier_freq * t_ref + phase_shifts[:, None])
        
        # 2. Correlation (Matrix Mul)
        # Signal(N, P) @ Refs.T(P, M) -> Scores(N, M)
        scores = np.dot(matrix, refs.T)
        
        # 3. Find Best Symbol
        best_symbols = np.argmax(scores, axis=1)
        
        # 4. Convert Symbols to Bits (Vectorized)
        # Ex: 2 (10) -> [1, 0]
        # Create bit masks: [2, 1] for k=2
        powers = 2 ** np.arange(k-1, -1, -1)
        # Broadcasting: (N, 1) & (1, k)
        decoded_bits = (best_symbols[:, None] & powers[None, :]) > 0
        
        return decoded_bits.flatten().astype(int)

    def modulate_mfsk(self, bits, M=4, baud_rate=1, base_freq=5, freq_sep=3, sampling_rate=1000):
        # Similar to BFSK but with M freqs
        bits = np.array(bits)
        k = int(np.log2(M))
        
        # Bits to Symbols
        rem = len(bits) % k
        if rem != 0: bits = np.concatenate([bits, np.zeros(k-rem, int)])
        symbols = (bits.reshape(-1, k) * (2**np.arange(k)[::-1])).sum(axis=1)
        
        # Symbols to Frequencies
        freqs = base_freq + symbols * freq_sep
        
        points_per_symbol = int(sampling_rate / baud_rate)
        
        # Vectorized Signal Gen
        # Expand freqs
        extended_freqs = np.repeat(freqs, points_per_symbol)
        t_local = np.tile(np.arange(points_per_symbol)/sampling_rate, len(symbols))
        
        signal = np.sin(2 * np.pi * extended_freqs * t_local)
        t_global = np.arange(len(signal)) / sampling_rate
        return t_global, signal

    def demodulate_mfsk(self, signal, M=4, baud_rate=1, base_freq=5, freq_sep=3, sampling_rate=1000):
        # Matrix Correlation Method
        points_per_symbol = int(sampling_rate / baud_rate)
        n_symbols = len(signal) // points_per_symbol
        
        matrix = signal[:n_symbols*points_per_symbol].reshape(n_symbols, points_per_symbol)
        
        # Ref Bank
        t_ref = np.arange(points_per_symbol) / sampling_rate
        # Calculate freqs for all M
        m_vals = np.arange(M)
        target_freqs = base_freq + m_vals * freq_sep
        # Create refs (M, points)
        refs = np.sin(2*np.pi * target_freqs[:, None] * t_ref)
        
        # Correlation
        scores = matrix @ refs.T
        best_symbols = np.argmax(scores, axis=1)
        
        # Symbols to bits
        k = int(np.log2(M))
        powers = 2 ** np.arange(k-1, -1, -1)
        decoded_bits = (best_symbols[:, None] & powers[None, :]) > 0
        return decoded_bits.flatten().astype(int)
    
    def modulate_dpsk(self, bits, baud_rate=1, carrier_freq=5, sampling_rate=1000):
        """
        AI OPTIMIZATION:
        DPSK is recursive (phase depends on prev).
        However, phase accumulation is just cumsum of phase changes!
        """
        bits = np.array(bits)
        # Bit 1 -> Phase change pi, Bit 0 -> No change
        phase_changes = bits * np.pi
        
        # Cumulative phase (Vectorized recursion!)
        phases = np.cumsum(phase_changes)
        
        points_per_bit = int(sampling_rate / baud_rate)
        extended_phases = np.repeat(phases, points_per_bit)
        
        t = np.arange(len(extended_phases)) / sampling_rate
        # Note: t is continuous here
        signal = np.sin(2 * np.pi * carrier_freq * t + extended_phases)
        
        return t, signal

    def demodulate_dpsk(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=1000):
        """
        GEMINI OPTIMIZATION (V2) - GÜNCELLENDİ:
        - Arkadaşının mantığına uygun olarak ilk bit için referans sinyal eklendi.
        - Böylece ilk bit atlanmadan doğru şekilde çözülür.
        """
        points_per_bit = int(sampling_rate / baud_rate)
        n_bits = len(signal) // points_per_bit
        
        # Sinyali matrise çevir: (N_bits, Points)
        matrix = signal[:n_bits*points_per_bit].reshape(n_bits, points_per_bit)
        
        # REFERANS SİNYAL OLUŞTURMA (Arkadaşının eklediği mantık)
        # İlk biti kıyaslamak için "öncesi" olarak temiz bir taşıyıcı (0 faz) yaratıyoruz.
        t_ref = np.arange(points_per_bit) / sampling_rate
        # Eğer carrier_freq parametresi None gelirse varsayılan 5 kullan (Güvenlik)
        cf = carrier_freq if carrier_freq is not None else 5
        reference_signal = np.sin(2 * np.pi * cf * t_ref)
        
        # MATRİS KAYDIRMA (VECTORIZED SHIFT)
        # prev dizisini oluştururken:
        # 1. matrix'i 1 aşağı kaydır.
        # 2. En başa (ilk satıra) az önce oluşturduğumuz referans sinyali koy.
        prev = np.roll(matrix, 1, axis=0)
        prev[0] = reference_signal 
        
        # KORELASYON (Dot Product)
        # Şu anki bit ile bir önceki (veya referans) arasındaki benzerlik
        corrs = np.sum(matrix * prev, axis=1)
        
        # Pozitif korelasyon -> Faz değişmemiş -> 0
        # Negatif korelasyon -> Faz değişmiş -> 1
        decoded_bits = np.where(corrs > 0, 0, 1)
        
        return decoded_bits
    
    # ==========================================
        # YENİ EKLENEN WRAPPER FONKSİYONLAR
        # (Arka planda optimize edilmiş demodulate_mpsk kullanırlar)
        # ==========================================

    def demodulate_bpsk(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """BPSK Wrapper (M=2)"""
        return self.demodulate_mpsk(signal, M=2, baud_rate=baud_rate, carrier_freq=carrier_freq, sampling_rate=sampling_rate)

    def demodulate_qpsk(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """QPSK Wrapper (M=4)"""
        return self.demodulate_mpsk(signal, M=4, baud_rate=baud_rate, carrier_freq=carrier_freq, sampling_rate=sampling_rate)

    def demodulate_8psk(self, signal, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """8-PSK Wrapper (M=8)"""
        return self.demodulate_mpsk(signal, M=8, baud_rate=baud_rate, carrier_freq=carrier_freq, sampling_rate=sampling_rate)