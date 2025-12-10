import numpy as np

def generate_digital_data(size=10):
    """Rastgele 0 ve 1'lerden oluşan bir dizi üretir."""
    return np.random.randint(0, 2, size)

def generate_analog_signal(duration=1.0, sampling_rate=1000, frequency=5):
    """Basit bir sinüs dalgası üretir (Analog mesaj sinyali)."""
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    return t, signal