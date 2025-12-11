import numpy as np

import numpy as np

class DigitalToDigital:
    def __init__(self):
        # Genlik (Voltaj) seviyesi
        self.amplitude = 5  # Örn: 5 Volt

    def encode_nrz_l(self, bits, baud_rate=1, sampling_rate=100):
        """
        NRZ-L Encoding:
        0 -> +V (Pozitif Genlik)
        1 -> -V (Negatif Genlik)
        
        Parametreler:
        - bits: 0 ve 1'lerden oluşan numpy dizisi veya listesi
        - baud_rate: Saniyede iletilen bit sayısı (varsayılan 1 bit/sn)
        - sampling_rate: Saniyedeki örnekleme sayısı (grafik çizimi için detay)
        """
        bits = np.array(bits)
        
        # 1. Adım: Her bitin ne kadar süreceğini hesapla
        bit_duration = 1.0 / baud_rate
        
        # 2. Adım: Her bit için kaç tane nokta (sample) oluşturacağımızı bul
        # Örn: 1 saniye sürüyorsa ve hızı 100 ise, her bit için 100 nokta üretmeliyiz.
        points_per_bit = int(sampling_rate * bit_duration)
        
        # 3. Adım: Sinyal dizisini oluştur
        signal = []
        
        for bit in bits:
            if bit == 0:
                # 0 ise: points_per_bit kadar +5V ekle
                signal_segment = np.ones(points_per_bit) * self.amplitude
            else:
                # 1 ise: points_per_bit kadar -5V ekle
                signal_segment = np.ones(points_per_bit) * -self.amplitude
            
            signal.extend(signal_segment)
            
        # 4. Adım: Zaman eksenini (x-axis) oluştur
        total_points = len(signal)
        # 0'dan başlayıp toplam süreye kadar giden zaman çizelgesi
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points)
        
        return time_axis, np.array(signal)


    def decode_nrz_l(self, signal, baud_rate=1, sampling_rate=100):
        """
        NRZ-L Decoding: Sinyal seviyesine bakarak bitleri geri kazanır.
        Mantık: Bir bit süresinin ortasındaki voltaja bak.
        Pozitifse -> 0
        Negatifse -> 1
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        decoded_bits = []
        
        # Sinyali bit uzunluklarına göre parça parça geziyoruz
        for i in range(0, len(signal), points_per_bit):
            # O anki bit'e ait sinyal parçasını al
            chunk = signal[i : i + points_per_bit]
            
            # Parçanın boş olup olmadığını kontrol et (son parça hatası olmasın)
            if len(chunk) == 0:
                break
                
            # Basit Yöntem: Parçanın ortasındaki değere bak
            # (Gürültülü ortamlarda ortalama almak daha iyidir ama şimdilik bu yeterli)
            mid_point_value = chunk[len(chunk) // 2]
            
            if mid_point_value > 0:
                decoded_bits.append(0)
            else:
                decoded_bits.append(1)
                
        return np.array(decoded_bits)



    def encode_bipolar_ami(self, bits, baud_rate=1, sampling_rate=100):
        """
        Bipolar-AMI Encoding:
        0 -> 0V
        1 -> Alternating +V and -V
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        signal = []
        
        # We need to remember the polarity of the previous '1'.
        # Let's start with -amplitude, so the first '1' becomes positive.
        last_one_voltage = -self.amplitude 
        
        for bit in bits:
            if bit == 0:
                # Logic 0 is represented by 0 Volts
                signal_segment = np.zeros(points_per_bit)
            else:
                # Logic 1 is represented by alternating voltage
                current_voltage = -last_one_voltage # Flip the polarity
                signal_segment = np.ones(points_per_bit) * current_voltage
                
                # Update memory for the next '1'
                last_one_voltage = current_voltage
            
            signal.extend(signal_segment)
            
        total_points = len(signal)
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points)
        
        return time_axis, np.array(signal)

    def decode_bipolar_ami(self, signal, baud_rate=1, sampling_rate=100):
        """
        Bipolar-AMI Decoding:
        If voltage is 0 -> Bit 0
        If voltage is non-zero (+V or -V) -> Bit 1
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        decoded_bits = []
        
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i : i + points_per_bit]
            
            if len(chunk) == 0:
                break
            
            # Check the middle point value
            mid_point_value = chunk[len(chunk) // 2]
            
            # In AMI, both +V and -V mean '1'. Only 0V means '0'.
            # We use a small threshold (e.g., 0.5) to account for floating point errors
            if abs(mid_point_value) < 0.5:
                decoded_bits.append(0)
            else:
                decoded_bits.append(1)
                
        return np.array(decoded_bits)



    def encode_manchester(self, bits, baud_rate=1, sampling_rate=100):
        """
        Manchester Encoding (IEEE 802.3 Standard):
        Bit 0: High -> Low transition (+V then -V)
        Bit 1: Low -> High transition (-V then +V)
        Each bit period is split into two halves.
        """
        bits = np.array(bits)
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        
        # Calculate split point (middle of the bit)
        half_points = points_per_bit // 2
        
        signal = []
        
        for bit in bits:
            if bit == 0:
                # 0: High to Low (+V, then -V)
                first_half = np.ones(half_points) * self.amplitude
                second_half = np.ones(points_per_bit - half_points) * -self.amplitude
            else:
                # 1: Low to High (-V, then +V)
                first_half = np.ones(half_points) * -self.amplitude
                second_half = np.ones(points_per_bit - half_points) * self.amplitude
            
            # Combine two halves for this single bit
            signal.extend(first_half)
            signal.extend(second_half)
            
        total_points = len(signal)
        time_axis = np.linspace(0, len(bits) * bit_duration, total_points)
        
        return time_axis, np.array(signal)

    def decode_manchester(self, signal, baud_rate=1, sampling_rate=100):
        """
        Manchester Decoding:
        We look at the transition.
        Instead of checking one point, we compare the first half and second half of the bit duration.
        """
        bit_duration = 1.0 / baud_rate
        points_per_bit = int(sampling_rate * bit_duration)
        half_points = points_per_bit // 2
        
        decoded_bits = []
        
        for i in range(0, len(signal), points_per_bit):
            chunk = signal[i : i + points_per_bit]
            
            if len(chunk) < points_per_bit:
                break
            
            # Sample logic:
            # Check a point in the first half (e.g., at 25% of the bit duration)
            # Check a point in the second half (e.g., at 75% of the bit duration)
            first_sample = chunk[half_points // 2]
            second_sample = chunk[half_points + (half_points // 2)]
            
            # Logic:
            # If First is Positive and Second is Negative -> High-to-Low -> Bit 0
            # If First is Negative and Second is Positive -> Low-to-High -> Bit 1
            
            if first_sample > 0 and second_sample < 0:
                decoded_bits.append(0)
            elif first_sample < 0 and second_sample > 0:
                decoded_bits.append(1)
            else:
                # Error state or noise (defaulting to 0 or handling error)
                decoded_bits.append(0) 
                
        return np.array(decoded_bits)


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
    
    dd = DigitalToDigital()
    
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
    
    # Run tests
    print("="*50)
    print("DIGITAL-TO-DIGITAL EXTENDED ENCODING TESTS")
    print("="*50)
    
    test_encoding("NRZ-L", dd.encode_nrz_l, dd.decode_nrz_l, test_bits)
    test_encoding("Bipolar-AMI", dd.encode_bipolar_ami, dd.decode_bipolar_ami, test_bits)
    test_encoding("Manchester", dd.encode_manchester, dd.decode_manchester, test_bits)
    test_encoding("NRZI", dd.encode_nrzi, dd.decode_nrzi, test_bits)
    test_encoding("Pseudoternary", dd.encode_pseudoternary, dd.decode_pseudoternary, test_bits)
    test_encoding("Differential Manchester (IEEE 802.5)", dd.encode_dif_manch, dd.decode_dif_manch, test_bits)
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED!")
    print("="*50)
