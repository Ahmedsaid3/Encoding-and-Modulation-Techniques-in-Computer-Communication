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