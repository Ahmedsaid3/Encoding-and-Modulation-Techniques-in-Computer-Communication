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
    
    def modulate_mfsk(self, bits, M=4, baud_rate=1, base_freq=5, freq_sep=3, sampling_rate=1000):
        bits = np.array(bits)
        k = int(np.log2(M))                # bits per symbol
        bit_duration = 1 / baud_rate
        points_per_symbol = int(bit_duration * sampling_rate)

        # bits → symbols
        symbols = []
        for i in range(0, len(bits), k):
            symbol = 0
            for b in bits[i:i+k]:
                symbol = (symbol << 1) | b
            symbols.append(symbol)

        signal = []

        for symbol in symbols:
            freq = base_freq + symbol * freq_sep
            t = np.linspace(0, bit_duration, points_per_symbol, endpoint=False)
            segment = np.sin(2 * np.pi * freq * t)
            signal.extend(segment)

        time_axis = np.linspace(0, len(symbols) * bit_duration, len(signal), endpoint=False)
        return time_axis, np.array(signal)
    
    def demodulate_mfsk(self, signal, M=4, baud_rate=1, base_freq=5, freq_sep=3, sampling_rate=1000):
        k = int(np.log2(M))
        bit_duration = 1 / baud_rate
        points_per_symbol = int(bit_duration * sampling_rate)

        # reference signals
        t = np.linspace(0, bit_duration, points_per_symbol, endpoint=False)
        references = [
            np.sin(2 * np.pi * (base_freq + i * freq_sep) * t)
            for i in range(M)
        ]

        decoded_bits = []

        for i in range(0, len(signal), points_per_symbol):
            chunk = signal[i:i+points_per_symbol]
            if len(chunk) < points_per_symbol:
                break

            # correlation
            scores = [np.sum(chunk * ref) for ref in references]
            symbol = np.argmax(scores)

            # symbol → bits
            for b in range(k-1, -1, -1):
                decoded_bits.append((symbol >> b) & 1)

        return np.array(decoded_bits)



    def modulate_dpsk(self, bits, baud_rate=1, carrier_freq=5, sampling_rate=1000):
        bits = np.array(bits)
        bit_duration = 1 / baud_rate
        points_per_bit = int(bit_duration * sampling_rate)

        current_phase = 0
        signal = []

        for bit in bits:
            if bit == 1:
                current_phase += np.pi   # phase flip

            t = np.linspace(0, bit_duration, points_per_bit, endpoint=False)
            segment = np.sin(2 * np.pi * carrier_freq * t + current_phase)
            signal.extend(segment)

        time_axis = np.linspace(0, len(bits) * bit_duration, len(signal), endpoint=False)
        return time_axis, np.array(signal)

    def demodulate_dpsk(self, signal, baud_rate=1, carrier_freq=None, sampling_rate=1000):
        bit_duration = 1 / baud_rate
        points_per_bit = int(bit_duration * sampling_rate)

        decoded_bits = []
        prev_chunk = None

        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i:i+points_per_bit]
            if len(chunk) < points_per_bit:
                break

            if prev_chunk is None:
                prev_chunk = chunk
                continue

            corr = np.sum(prev_chunk * chunk)

            # same phase → 0, flipped → 1
            decoded_bits.append(0 if corr > 0 else 1)
            prev_chunk = chunk

        return np.array(decoded_bits)


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
    

    def modulate_bfsk(self, bits, baud_rate=1, freq_0=5, freq_1=10, sampling_rate=100):
        """
        Binary FSK (Frequency Shift Keying) Modulation.
        Bit 0 -> Carrier with frequency freq_0
        Bit 1 -> Carrier with frequency freq_1
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        total_duration = len(bits) * bit_duration
        total_points = int(total_duration * sampling_rate)
        
        time_axis = np.linspace(0, total_duration, total_points, endpoint=False)
        
        # Strategy: Create two full carrier waves for the entire duration
        carrier_signal_0 = np.sin(2 * np.pi * freq_0 * time_axis)
        carrier_signal_1 = np.sin(2 * np.pi * freq_1 * time_axis)
        
        # Create masks
        mask_0 = np.zeros_like(time_axis)
        mask_1 = np.zeros_like(time_axis)
        
        points_per_bit = int(sampling_rate * bit_duration)
        
        for i, bit in enumerate(bits):
            start = i * points_per_bit
            end = (i + 1) * points_per_bit
            
            if bit == 0:
                mask_0[start:end] = 1
                mask_1[start:end] = 0
            else:
                mask_0[start:end] = 0
                mask_1[start:end] = 1
                
        # Combine: Select the active frequency for each segment
        # Signal = (Mask0 * Carrier0) + (Mask1 * Carrier1)
        fsk_signal = (mask_0 * carrier_signal_0 * self.amplitude) + \
                     (mask_1 * carrier_signal_1 * self.amplitude)
                     
        return time_axis, fsk_signal

    def demodulate_bfsk(self, signal, baud_rate=1, freq_0=5, freq_1=10, sampling_rate=100):
        """
        FSK Demodulation using Correlation (Matched Filter concept).
        We compare the incoming chunk against local references of freq_0 and freq_1.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        demodulated_bits = []
        
        # Generate local reference carriers for ONE bit duration
        t_ref = np.linspace(0, bit_duration, points_per_bit, endpoint=False)
        ref_signal_0 = np.sin(2 * np.pi * freq_0 * t_ref)
        ref_signal_1 = np.sin(2 * np.pi * freq_1 * t_ref)
        
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i : i + points_per_bit]
            
            if len(chunk) < points_per_bit:
                break
            
            # Correlation: Multiply chunk by reference and Sum
            # If the chunk contains freq_0, correlation with ref_signal_0 will be high.
            # If the chunk contains freq_1, correlation with ref_signal_0 will be low (near zero).
            score_0 = np.sum(chunk * ref_signal_0)
            score_1 = np.sum(chunk * ref_signal_1)
            
            # Compare scores
            if score_1 > score_0:
                demodulated_bits.append(1)
            else:
                demodulated_bits.append(0)
                
        return np.array(demodulated_bits)


    def modulate_mpsk(self, bits, M=2, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        M-ary Phase Shift Keying (MPSK) Modulation.
        Can function as BPSK (M=2), QPSK (M=4), 8-PSK (M=8), etc.
        
        Args:
            M: Number of phases (Must be a power of 2: 2, 4, 8, 16...)
            baud_rate: Symbols per second (Not bits per second!)
        """
        bits = np.array(bits)
        
        # 1. Calculate bits per symbol (k)
        # For M=2 -> k=1, M=4 -> k=2, M=8 -> k=3
        k = int(np.log2(M))
        
        # Ensure input bit length is divisible by k (pad with 0s if needed)
        remainder = len(bits) % k
        if remainder != 0:
            padding = np.zeros(k - remainder, dtype=int)
            bits = np.concatenate([bits, padding])
            
        # 2. Group bits into symbols
        # Reshape bits into a matrix of rows with k items
        reshaped_bits = bits.reshape(-1, k)
        
        # Convert binary rows to integer values (0 to M-1)
        # Example: [1, 0] -> 2
        # We use powers of 2: [1, 2, 4...] logic
        powers_of_two = 2 ** np.arange(k)[::-1]
        symbols = (reshaped_bits * powers_of_two).sum(axis=1)
        
        # 3. Generate Signal
        symbol_duration = 1.0 / baud_rate
        points_per_symbol = int(sampling_rate * symbol_duration)
        mpsk_signal = []
        
        # We need a continuous time axis for the plot
        total_symbols = len(symbols)
        
        for i, sym_val in enumerate(symbols):
            # Calculate Phase Shift for this symbol
            # Phase = symbol_val * (2*pi / M)
            phase_shift = sym_val * (2 * np.pi / M)
            
            # Generate time for this specific symbol slot
            # Note: We use local time t (0 to duration) for the sine wave 
            # to make the phase shift clearly visible relative to t=0 base.
            t = np.linspace(0, symbol_duration, points_per_symbol, endpoint=False)
            
            # Waveform: A * sin(2*pi*f*t + phase)
            segment = self.amplitude * np.sin(2 * np.pi * carrier_freq * t + phase_shift)
            
            mpsk_signal.extend(segment)
            
        total_points = len(mpsk_signal)
        # Create a full time axis for return
        time_axis = np.linspace(0, total_symbols * symbol_duration, total_points, endpoint=False)
        
        return time_axis, np.array(mpsk_signal)

    def demodulate_mpsk(self, signal, M=2, baud_rate=1, carrier_freq=5, sampling_rate=100):
        """
        MPSK Demodulation using Coherent Detection (Correlation with Reference Phases).
        """
        k = int(np.log2(M))
        symbol_duration = 1.0 / baud_rate
        points_per_symbol = int(sampling_rate * symbol_duration)
        
        demodulated_bits = []
        
        # Pre-calculate ALL possible reference signals (0 to M-1)
        references = []
        t_ref = np.linspace(0, symbol_duration, points_per_symbol, endpoint=False)
        
        for sym_val in range(M):
            phase_shift = sym_val * (2 * np.pi / M)
            ref_sig = np.sin(2 * np.pi * carrier_freq * t_ref + phase_shift)
            references.append(ref_sig)
            
        # Loop through the signal symbol by symbol
        for i in range(0, len(signal), points_per_symbol):
            chunk = signal[i : i + points_per_symbol]
            
            if len(chunk) < points_per_symbol:
                break
            
            # Find which reference correlates best with the chunk
            best_correlation = -float('inf')
            best_symbol = 0
            
            for sym_val, ref_sig in enumerate(references):
                # Correlation = sum(chunk * ref)
                correlation = np.sum(chunk * ref_sig)
                
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_symbol = sym_val
            
            # Convert the best symbol (int) back to bits (binary)
            # Example: 2 -> [1, 0] (for k=2)
            # Using bitwise operations to extract bits
            binary_string = format(best_symbol, f'0{k}b') # e.g., '10'
            symbol_bits = [int(b) for b in binary_string]
            
            demodulated_bits.extend(symbol_bits)
            
        return np.array(demodulated_bits)



# TEST ALANI 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sınıf örneğini oluşturuyoruz
    da = DigitalToAnalog()
    
    def test_da_modulation(name, mod_func, demod_func, test_bits, drop_first=False, **kwargs):
        """
        Digital-to-Analog modülasyonları test etmek için genel yardımcı fonksiyon.
        **kwargs: baud_rate, carrier_freq, freq_0, freq_1 gibi parametreleri dinamik alır.
        """
        print(f"\n{'='*60}")
        print(f"Testing {name}...")
        print(f"Original Bits:  {test_bits}")
        
        # 1. Modülasyon
        # kwargs içindeki parametreleri (frekans vb.) fonksiyona iletiyoruz
        t, s = mod_func(test_bits, **kwargs)
        
        # 2. Demodülasyon
        recovered_bits = demod_func(s, **kwargs)
        print(f"Recovered Bits: {list(recovered_bits)}")
        
        # 3.0 DPSK için hazırlık
        expected = test_bits[1:] if drop_first else test_bits

        if list(recovered_bits[:len(expected)]) == list(expected):
            print(f"RESULT: SUCCESS! {name} works correctly.")
        else:
            print(f"RESULT: FAILURE! Data mismatch.")

        # 4. Görselleştirme
        plt.figure(figsize=(12, 4))
        # Analog sinyal olduğu için 'step' yerine 'plot' kullanıyoruz
        plt.plot(t, s, linewidth=1.5)
        plt.title(f"{name} Modulation")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True, alpha=0.3)
        
        # Bit sınırlarını çiz (Görsellik için)
        # kwargs içinde baud_rate varsa onu al, yoksa 1 kabul et
        baud_rate = kwargs.get('baud_rate', 1)
        bit_duration = 1.0 / baud_rate
        total_time = len(test_bits) * bit_duration
        
        # Dikey çizgilerle bit aralıklarını göster
        for x in np.arange(0, total_time + bit_duration, bit_duration):
            plt.axvline(x, color='red', linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.show()

    # --- TEST VERİLERİ ---
    test_bits = [0, 1, 0, 1, 1, 1, 1, 0]
    
    print("="*60)
    print("DIGITAL-TO-ANALOG MODULATION TESTS")
    print("="*60)
    
    # 1. TEST: ASK (Amplitude Shift Keying)
    # Parametreleri buraya sözlük gibi giriyoruz
    test_da_modulation(
        name="ASK (Amplitude Shift Keying)", 
        mod_func=da.modulate_ask, 
        demod_func=da.demodulate_ask, 
        test_bits=test_bits,
        baud_rate=1, 
        carrier_freq=4,  # Her bit içine 4 tam dalga sığsın
        sampling_rate=1000
    )
    
    # 2. TEST: FSK (Frequency Shift Keying)
    test_da_modulation(
        name="FSK (Frequency Shift Keying)", 
        mod_func=da.modulate_bfsk, 
        demod_func=da.demodulate_bfsk, 
        test_bits=test_bits,
        baud_rate=1,
        freq_0=4,   # 0 biti için 4 Hz
        freq_1=8,   # 1 biti için 8 Hz
        sampling_rate=1000
    )
    

    # 3. TEST: QPSK (M=4) -> 2 bits per symbol
    # Notice: baud_rate is symbol rate. 
    # If baud_rate=1 and M=4, we are sending 2 bits per second.
    test_da_modulation(
        name="QPSK (M=4, Phase Shift Keying)", 
        mod_func=da.modulate_mpsk, 
        demod_func=da.demodulate_mpsk, 
        test_bits=test_bits,
        M=4,        # QPSK
        baud_rate=1,
        carrier_freq=2, # Keep freq low to see phase shifts easily
        sampling_rate=1000
    )

    # 4. TEST: BPSK (M=2) -> 1 bit per symbol (Standard PSK)
    test_da_modulation(
        name="BPSK (M=2)", 
        mod_func=da.modulate_mpsk, 
        demod_func=da.demodulate_mpsk, 
        test_bits=test_bits,
        M=2,
        baud_rate=1,
        carrier_freq=2,
        sampling_rate=1000
    )

    # 5. TEST: MPSK (M=8) -> 3 bits per symbol (8-PSK)
    test_da_modulation(
        name="MPSK (M=8)", 
        mod_func=da.modulate_mpsk, 
        demod_func=da.demodulate_mpsk, 
        test_bits=test_bits,
        M=8,
        baud_rate=1,
        carrier_freq=2,
        sampling_rate=1000
    )

    # 6. TEST: MFSK (M-ary FSK)
    test_da_modulation(
        name="MFSK (M=4)", 
        mod_func=da.modulate_mfsk, 
        demod_func=da.demodulate_mfsk, 
        test_bits=test_bits,
        M=4,               # 4-FSK → 2 bits per symbol
        baud_rate=1,
        base_freq=3,       # starting frequency
        freq_sep=3,        # spacing between tones
        sampling_rate=1000
    )


    test_da_modulation(
        name="DPSK (Differential PSK)", 
        mod_func=da.modulate_dpsk, 
        demod_func=da.demodulate_dpsk, 
        test_bits=test_bits,
        drop_first=True,         
        baud_rate=1,
        carrier_freq=2,
        sampling_rate=1000
    )


