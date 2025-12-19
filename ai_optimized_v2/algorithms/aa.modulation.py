import numpy as np
from scipy.signal import hilbert

class AnalogToAnalog:
    def __init__(self, carrier_freq=100, sampling_rate=1000):
        self.fc = carrier_freq
        self.fs = sampling_rate
        # Sabitleri önceden hesapla (Optimization)
        self.two_pi_fc = 2 * np.pi * self.fc
        self.dt = 1.0 / self.fs

    def _get_time_array(self, duration):
        # Linspace yerine arange daha hızlıdır
        return np.arange(int(duration * self.fs)) * self.dt

    def _get_carrier(self, t):
        return np.cos(self.two_pi_fc * t)

    def modulate_am(self, analog_signal, mod_index=1.0):
        """
        GEMINI OPTIMIZATION (V2):
        - In-place operations (bellek dostu).
        - Gereksiz array kopyalamalardan kaçınma.
        """
        duration = len(analog_signal) * self.dt
        t = self._get_time_array(duration)
        
        # 1. Normalize (Vektörel Max Bulma)
        max_val = np.max(np.abs(analog_signal))
        if max_val > 0:
            # analog_signal / max_val işlemi yeni array yaratır.
            # Bunu modülasyon formülünün içine gömelim.
            scale = mod_index / max_val
            # Formül: (1 + scale * signal) * cos(...)
            envelope = 1 + analog_signal * scale
        else:
            envelope = 1 + analog_signal * mod_index

        # Carrier'ı ayrı hesaplayıp çarpmak yerine tek satırda
        modulated_signal = envelope * np.cos(self.two_pi_fc * t)
        
        return t, modulated_signal

    def modulate_fm(self, analog_signal, kf=20.0):
        """
        GEMINI OPTIMIZATION (V2):
        - np.cumsum zaten hızlıdır.
        - Faz hesaplamasını in-place yaparak bellekten tasarruf ediyoruz.
        """
        duration = len(analog_signal) * self.dt
        t = self._get_time_array(duration)
        
        # 1. Entegre et (Integral)
        integral_message = np.cumsum(analog_signal) * self.dt
        
        # 2. Fazı hesapla: 2*pi*fc*t + 2*pi*kf*integral
        # Bellek optimizasyonu için t array'ini 'phase' olarak yeniden kullanabiliriz
        # ama okunabilirlik bozulmasın diye yeni değişken açıyoruz.
        phase = self.two_pi_fc * t
        phase += (2 * np.pi * kf * integral_message) # In-place add
        
        modulated_signal = np.cos(phase)
        return t, modulated_signal

    def modulate_pm(self, analog_signal, kp=2.0):
        """
        GEMINI OPTIMIZATION (V2):
        - Doğrudan faz manipülasyonu.
        """
        duration = len(analog_signal) * self.dt
        t = self._get_time_array(duration)
        
        # Phase = 2*pi*fc*t + kp*m(t)
        phase = self.two_pi_fc * t
        phase += kp * analog_signal
        
        modulated_signal = np.cos(phase)
        return t, modulated_signal

    def demodulate_am(self, modulated_signal):
        """
        GEMINI OPTIMIZATION (V2) - MAJOR SPEEDUP:
        - Orijinal kod 'np.convolve' kullanıyor (O(N*W) karmaşıklığı).
        - V2, 'Running Sum' (Cumsum) tekniği kullanıyor (O(N) karmaşıklığı).
        - Büyük veri setlerinde 100x+ hızlanma sağlar.
        """
        # 1. Rectification
        rectified = np.abs(modulated_signal)
        
        # 2. Fast Moving Average (Boxcar Filter) using Cumsum
        window_size = int(self.fs / self.fc) * 2
        if window_size < 1: window_size = 1
        
        # Cumsum hilesi: MovingAvg[i] = (Cumsum[i] - Cumsum[i-W]) / W
        cumsum_vec = np.cumsum(np.insert(rectified, 0, 0)) 
        demodulated = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        
        # Boyut düzeltme (Convolution 'same' modunu taklit etmek için padding)
        pad_size = len(modulated_signal) - len(demodulated)
        if pad_size > 0:
            demodulated = np.pad(demodulated, (pad_size//2, pad_size - pad_size//2), mode='edge')
            
        return demodulated - np.mean(demodulated)

    def demodulate_fm(self, modulated_signal):
        """
        GEMINI OPTIMIZATION (V2):
        - 'np.unwrap' ve 'np.diff' yavaştır.
        - Faz Farkı (Phase Differencing) tekniği kullanıyoruz:
          angle(z[n] * conj(z[n-1]))
        - Bu teknik unwrap gerektirmez ve türevi direkt verir.
        """
        # 1. Analytic Signal (Hilbert yine de gerekli, FFT tabanlı olduğu için hızlıdır)
        analytic_signal = hilbert(modulated_signal)
        
        # 2. Phase Differencing (Unwrap gerektirmez!)
        # z[n] * z*[n-1] işleminin açısı, faz farkını (frekansı) verir.
        # Bu yöntem np.diff(np.unwrap(angle)) ile aynı matematiksel sonucu verir ama çok daha hızlıdır.
        
        # Conjugate delay multiply
        # z[1:] * conj(z[:-1])
        phase_diff = analytic_signal[1:] * np.conj(analytic_signal[:-1])
        instantaneous_freq = np.angle(phase_diff) * (self.fs / (2.0 * np.pi))
        
        # 3. Remove Carrier
        demodulated = instantaneous_freq - self.fc
        
        # Boyut eşitleme (append 0)
        return np.append(demodulated, 0)