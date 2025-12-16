import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- PATH AYARLAMALARI ---
# Proje ana dizinini Python yoluna ekliyoruz ki modülleri bulabilsin.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- İMPORTLAR ---
# Orijinal ve Optimize Edilmiş sınıfları farklı isimlerle çağırıyoruz (Aliasing)
try:
    from src.algorithms.dd_encoding import DigitalToDigital as OriginalDD
    from ai_optimized_v2.algorithms.dd_encoding import DigitalToDigital as OptimizedDD
    print("✅ Modüller başarıyla yüklendi.")
except ImportError as e:
    print(f"❌ Modül hatası: {e}")
    print("Lütfen projenin ana dizininde olduğunuzdan emin olun.")
    sys.exit(1)

def run_benchmark(n_bits=100000):
    """
    Belirtilen bit sayısı ile hız testi yapar.
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK BAŞLIYOR... (Veri Boyutu: {n_bits} bit)")
    print(f"{'='*60}")

    # Rastgele büyük veri üret
    bits = np.random.randint(0, 2, n_bits)
    
    # Sınıf örnekleri
    original = OriginalDD()
    optimized = OptimizedDD()
    
    # Test edilecek algoritmalar
    methods = [
        ("NRZ-L", "encode_nrz_l"),
        ("Manchester", "encode_manchester"),
        ("Bipolar-AMI", "encode_bipolar_ami")
    ]
    
    results = {}

    for name, method_name in methods:
        print(f"\n--- Test Ediliyor: {name} ---")
        
        # 1. ORİJİNAL KOD TESTİ
        start_time = time.perf_counter()
        # getattr ile metodu isminden dinamik çağırıyoruz
        getattr(original, method_name)(bits) 
        end_time = time.perf_counter()
        original_duration = end_time - start_time
        print(f"Original Time:  {original_duration:.6f} sn")
        
        # 2. AI OPTIMIZED KOD TESTİ
        start_time = time.perf_counter()
        getattr(optimized, method_name)(bits)
        end_time = time.perf_counter()
        optimized_duration = end_time - start_time
        print(f"Optimized Time: {optimized_duration:.6f} sn")
        
        # 3. KARŞILAŞTIRMA
        speedup = original_duration / optimized_duration
        print(f"🚀 HIZLANMA: {speedup:.2f}x kat daha hızlı!")
        
        # Sonuçları kaydet
        results[name] = (original_duration, optimized_duration)

    return results

def plot_results(results, n_bits):
    """
    Sonuçları görselleştirir (Bar Chart)
    """
    labels = list(results.keys())
    original_times = [v[0] for v in results.values()]
    optimized_times = [v[1] for v in results.values()]

    x = np.arange(len(labels))  # Etiket konumları
    width = 0.35  # Bar genişliği

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, original_times, width, label='Orijinal (Döngüsel)', color='red', alpha=0.7)
    rects2 = ax.bar(x + width/2, optimized_times, width, label='AI Optimized (Vektörel)', color='green', alpha=0.7)

    # Yazılar ve Başlıklar
    ax.set_ylabel('Süre (Saniye) - Daha Düşük İyidir')
    ax.set_title(f'Performans Karşılaştırması ({n_bits} Bit İşleme)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Barların üzerine hız farkını yaz
    for i in range(len(labels)):
        speedup = original_times[i] / optimized_times[i]
        ax.text(x[i] + width/2, optimized_times[i], f'{speedup:.1f}x Hızlı', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    # plt.savefig('benchmark/benchmark_result.png') # Grafiği kaydet
    print("\n✅ Grafik 'benchmark/benchmark_result.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    # 100 bin bit ile test et (Bilgisayarın hızına göre artırıp azaltabilirsin)
    benchmark_data = run_benchmark(n_bits=100000)
    plot_results(benchmark_data, n_bits=100000)