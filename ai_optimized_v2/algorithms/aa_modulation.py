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

    def modulate_am(self, analog_signal, mod_index=1.0):
        """
        GEMINI OPTIMIZATION (V2):
        - In-place operations (bellek dostu).
        """
        duration = len(analog_signal) * self.dt
        t = self._get_time_array(duration)
        
        # 1. Normalize (Vektörel Max Bulma)
        max_val = np.max(np.abs(analog_signal))
        if max_val > 0:
            scale = mod_index / max_val
            envelope = 1 + analog_signal * scale
        else:
            envelope = 1 + analog_signal * mod_index

        modulated_signal = envelope * np.cos(self.two_pi_fc * t)
        return t, modulated_signal

    def modulate_fm(self, analog_signal, kf=20.0):
        """
        GEMINI OPTIMIZATION (V2):
        - np.cumsum zaten hızlıdır.
        """
        duration = len(analog_signal) * self.dt
        t = self._get_time_array(duration)
        
        integral_message = np.cumsum(analog_signal) * self.dt
        
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
        
        phase = self.two_pi_fc * t
        phase += kp * analog_signal
        
        modulated_signal = np.cos(phase)
        return t, modulated_signal

    def demodulate_am(self, modulated_signal):
        """
        GEMINI OPTIMIZATION (V2) - MAJOR SPEEDUP:
        - 'Running Sum' (Cumsum) tekniği ile O(N) filtreleme.
        """
        rectified = np.abs(modulated_signal)
        
        window_size = int(self.fs / self.fc) * 2
        if window_size < 1: window_size = 1
        
        # Cumsum hilesi: MovingAvg
        cumsum_vec = np.cumsum(np.insert(rectified, 0, 0)) 
        demodulated = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
        
        # Boyut düzeltme
        pad_size = len(modulated_signal) - len(demodulated)
        if pad_size > 0:
            demodulated = np.pad(demodulated, (pad_size//2, pad_size - pad_size//2), mode='edge')
            
        return demodulated - np.mean(demodulated)

    def demodulate_fm(self, modulated_signal):
        """
        GEMINI OPTIMIZATION (V2):
        - Phase Differencing (Unwrap gerektirmez, çok hızlı).
        """
        analytic_signal = hilbert(modulated_signal)
        
        # z[n] * z*[n-1] işleminin açısı, faz farkını (frekansı) verir.
        phase_diff = analytic_signal[1:] * np.conj(analytic_signal[:-1])
        instantaneous_freq = np.angle(phase_diff) * (self.fs / (2.0 * np.pi))
        
        demodulated = instantaneous_freq - self.fc
        return np.append(demodulated, 0)

    def demodulate_pm(self, modulated_signal, kp=2.0):
        """
        GEMINI OPTIMIZATION (V2):
        - Down-conversion (Heterodyning) Yöntemi.
        - Fazı çıkarmak yerine, komple sinyali Baseband'e indiriyoruz.
        - Bu yöntem 'np.unwrap' hatalarını azaltır ve matematiksel olarak daha sağlamdır.
        """
        # 1. Analytic Signal
        analytic_signal = hilbert(modulated_signal)
        
        # 2. Down-conversion (Carrier Removal in Complex Domain)
        # Sinyali e^(-j*wc*t) ile çarparak frekansı 0'a kaydırıyoruz.
        # Bu işlem taşıyıcıyı matematiksel olarak "siler".
        duration = len(modulated_signal) * self.dt
        t = self._get_time_array(duration) # t arrayini yeniden oluşturuyoruz (veya parametre alabiliriz)
        
        # Downconverter vektörü: e^(-j * 2*pi*fc * t)
        downconverter = np.exp(-1j * self.two_pi_fc * t)
        
        # Baseband sinyal (Taşıyıcısız, sadece mesajın fazı kaldı)
        baseband_signal = analytic_signal * downconverter
        
        # 3. Angle Extraction & Unwrapping
        # Artık açıyı aldığımızda doğrudan kp*m(t)'yi elde ederiz.
        demodulated_phase = np.unwrap(np.angle(baseband_signal))
        
        # 4. Scaling
        # Remove DC offset ve kp'ye böl
        demodulated = (demodulated_phase - np.mean(demodulated_phase)) / kp
        
        return demodulated