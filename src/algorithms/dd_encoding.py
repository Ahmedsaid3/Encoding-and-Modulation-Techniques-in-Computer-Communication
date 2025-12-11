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






# --- TEST ALANI (Sadece bu dosya çalışınca çalışır) ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    dd = DigitalToDigital()
    
    # 1. Orijinal Veri
    original_bits = [0, 1, 0, 0, 1, 1, 0]
    print(f"Gönderilen Bitler: {original_bits}")
    
    # 2. Encoding (Sinyale Çevir)
    t, s = dd.encode_nrz_l(original_bits)
    
    # 3. Decoding (Geri Çevir)
    recovered_bits = dd.decode_nrz_l(s)
    print(f"Alınan Bitler:     {list(recovered_bits)}")
    
    # Doğrulama
    if list(original_bits) == list(recovered_bits):
        print("SONUÇ: BAŞARILI! Veri kayıpsız iletildi.")
    else:
        print("SONUÇ: HATA! Veri bozuldu.")

    # Görselleştirme
    plt.step(t, s, where='post')
    plt.title("NRZ-L Encoding")
    plt.grid(True)
    plt.ylim(-6, 6)
    plt.show()

    # BIPOLAR AMI
    # Test Data
    original_bits = [0, 1, 0, 1, 1, 0, 0, 1]
    print(f"Original Bits: {original_bits}")

    # --- TEST BIPOLAR AMI ---
    print("\nTesting Bipolar-AMI...")
    t, s = dd.encode_bipolar_ami(original_bits)
    
    recovered_bits = dd.decode_bipolar_ami(s)
    print(f"Recovered Bits: {list(recovered_bits)}")
    
    if list(original_bits) == list(recovered_bits):
        print("RESULT: SUCCESS! Bipolar-AMI works correctly.")
    else:
        print("RESULT: FAILURE! Data mismatch.")

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.step(t, s, where='post', linewidth=2)
    plt.title("Bipolar-AMI Encoding")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.grid(True)
    plt.ylim(-6, 6)
    
    # Add horizontal line at 0 for clarity
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()


    # MANCHESTER TESTI 
    # Test Data
    original_bits = [0, 1, 0, 0, 1, 1, 0]
    print(f"Original Bits: {original_bits}")

    # --- TEST MANCHESTER ---
    print("\nTesting Manchester...")
    t, s = dd.encode_manchester(original_bits)
    
    recovered_bits = dd.decode_manchester(s)
    print(f"Recovered Bits: {list(recovered_bits)}")
    
    if list(original_bits) == list(recovered_bits):
        print("RESULT: SUCCESS! Manchester works correctly.")
    else:
        print("RESULT: FAILURE! Data mismatch.")

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.step(t, s, where='post', linewidth=2)
    plt.title("Manchester Encoding (IEEE 802.3)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.grid(True)
    plt.ylim(-6, 6)
    
    # Grid çizgilerini bit aralıklarına göre ayarlarsak geçişleri daha iyi görürüz
    # Her 1 saniyede bir dikey çizgi
    import math
    for x in range(len(original_bits) + 1):
        plt.axvline(x, color='gray', linestyle='--', alpha=0.5)
        
    plt.show()