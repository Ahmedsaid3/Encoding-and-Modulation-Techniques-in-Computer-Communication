import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- PATH AYARLARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- İMPORTLAR ---
try:
    # 1. ORİJİNAL (Referans)
    from src.algorithms.dd_encoding import DigitalToDigital as OrigDD
    from src.algorithms.da_modulation import DigitalToAnalog as OrigDA
    from src.algorithms.ad_encoding import AnalogToDigital as OrigAD
    
    # 2. V1 (ARKADAŞ - ChatGPT)
    from ai_optimized_v1.algorithms.dd_encoding import DigitalToDigital as FriendDD
    from ai_optimized_v1.algorithms.da_modulation import DigitalToAnalog as FriendDA
    from ai_optimized_v1.algorithms.ad_encoding import AnalogToDigital as FriendAD
    
    # 3. V2 (GEMINI - Sen)
    from ai_optimized_v2.algorithms.dd_encoding import DigitalToDigital as GeminiDD
    from ai_optimized_v2.algorithms.da_modulation import DigitalToAnalog as GeminiDA
    from ai_optimized_v2.algorithms.ad_encoding import AnalogToDigital as GeminiAD

    print("✅ Tüm Modüller (Original, V1, V2) başarıyla yüklendi.")

except ImportError as e:
    print(f"❌ Modül hatası: {e}")
    print("⚠️ Lütfen 'ai_optimized_v1' ve 'ai_optimized_v2' klasörlerinin dolu olduğundan emin olun.")
    sys.exit(1)

def run_category_benchmark(category_name, tests, n_bits):
    """
    Belirli bir kategori (DD, DA veya AD) için testleri çalıştırır ve sonuçları döndürür.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK KATEGORİSİ: {category_name} (Veri Boyutu: {n_bits} bit)")
    print(f"{'='*70}")

    # Test verilerini hazırla
    bits = np.random.randint(0, 2, n_bits)
    # Analog sinyal (PCM/Delta için)
    t = np.linspace(0, 1, n_bits)
    analog_signal = np.sin(2 * np.pi * 5 * t)
    
    # Nesneleri başlat (Her kategori için ayrı ayrı)
    instances = {
        "DD": {
            "Orig": OrigDD(), "V1": FriendDD(), "V2": GeminiDD()
        },
        "DA": {
            "Orig": OrigDA(), "V1": FriendDA(), "V2": GeminiDA()
        },
        "AD": {
            "Orig": OrigAD(), "V1": FriendAD(), "V2": GeminiAD()
        }
    }
    
    current_instances = instances[category_name] # Sadece ilgili kategoriyi al
    results = {}

    for test_label, method_name, input_type, kwargs in tests:
        print(f"--- Test: {test_label} ---")
        
        # Girdi argümanını belirle
        if input_type == "bits":
            args = [bits]
        elif input_type == "analog":
            args = [analog_signal]
        elif input_type == "signal_from_mod": 
            # Demodülasyon testi için önce modüle edilmiş sinyal lazım
            # Orijinal sınıfı kullanarak sinyali üretelim (adil olsun)
            temp_mod_method = method_name.replace("demodulate", "modulate").replace("decode", "encode")
            if "pcm" in method_name: temp_mod_method = "encode_pcm"
            
            # Modülasyonu çalıştırıp sinyali al
            if "pcm" in method_name:
                signal_arg = current_instances["Orig"].encode_pcm(analog_signal, **kwargs)[0] # bits döner
            else:
                # Modülasyon genelde (time, signal) döner, [1] ile sinyali alıyoruz
                signal_arg = getattr(current_instances["Orig"], temp_mod_method)(bits, **kwargs)[1]
            args = [signal_arg]

        times = []
        for version_name in ["Orig", "V1", "V2"]:
            obj = current_instances[version_name]
            
            try:
                start = time.perf_counter()
                getattr(obj, method_name)(*args, **kwargs)
                duration = time.perf_counter() - start
            except Exception as e:
                print(f"  ⚠️ {version_name} Hata: {e}")
                duration = 0 # Hata durumunda 0
            
            times.append(duration)

        t_orig, t_v1, t_v2 = times
        
        # Hızlanma Yazdır
        s_v1 = t_orig / t_v1 if t_v1 > 1e-9 else 0
        s_v2 = t_orig / t_v2 if t_v2 > 1e-9 else 0
        
        print(f"  ⏱️  Orig: {t_orig:.4f}s | V1: {t_v1:.4f}s | V2: {t_v2:.4f}s")
        print(f"  🚀 V1 Hız: {s_v1:.1f}x | V2 Hız: {s_v2:.1f}x")
        
        results[test_label] = times
    
    return results

def plot_category_results(results, category_name, filename):
    labels = list(results.keys())
    t_orig = [v[0] for v in results.values()]
    t_v1 = [v[1] for v in results.values()]
    t_v2 = [v[2] for v in results.values()]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    
    rects1 = ax.bar(x - width, t_orig, width, label='Original', color='#ff7675')
    rects2 = ax.bar(x,        t_v1,   width, label='V1 (Friend/ChatGPT)', color='#74b9ff')
    rects3 = ax.bar(x + width, t_v2,   width, label='V2 (Sen/Gemini)', color='#55efc4')

    ax.set_ylabel('Çalışma Süresi (Saniye) - Düşük Daha İyi')
    ax.set_title(f'{category_name} Performans Kıyaslaması')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Sadece V2'nin üzerine hızlanma katını yazalım (Okunabilirlik için)
    for i in range(len(labels)):
        if t_v2[i] > 1e-9:
            speedup = t_orig[i] / t_v2[i]
            # Eğer hızlanma çok büyükse, barın üstüne sığmayabilir, biraz yukarı yazalım
            ax.text(x[i] + width, t_v2[i], f'{speedup:.0f}x', 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color='#006266')

    plt.tight_layout()
    # save_path = f'benchmark/{filename}'
    # plt.savefig(save_path)
    # print(f"✅ Grafik kaydedildi: {save_path}")
    plt.show() # İstersen açabilirsin

if __name__ == "__main__":
    # Veri boyutu (Load Test için yüksek tutuyoruz)
    N_BITS = 50000 
    
    # --- 1. Digital-to-Digital (DD) Testleri ---
    dd_tests = [
        ("NRZ-L", "encode_nrz_l", "bits", {}),
        ("Bipolar-AMI", "encode_bipolar_ami", "bits", {}),
        ("Manchester", "encode_manchester", "bits", {}),
        ("NRZI", "encode_nrzi", "bits", {}),
        ("Pseudoternary", "encode_pseudoternary", "bits", {}),
        ("Diff. Manch.", "encode_dif_manch", "bits", {})
    ]
    results_dd = run_category_benchmark("DD", dd_tests, N_BITS)
    plot_category_results(results_dd, "Digital-to-Digital Benchmark", "benchmark_dd.png")

    # --- 2. Digital-to-Analog (DA) Testleri ---
    da_tests = [
        ("ASK Mod", "modulate_ask", "bits", {}),
        ("ASK Demod", "demodulate_ask", "signal_from_mod", {}),
        ("BFSK Mod", "modulate_bfsk", "bits", {}),
        ("BFSK Demod", "demodulate_bfsk", "signal_from_mod", {}),
        ("BPSK Mod", "modulate_mpsk", "bits", {"M": 2}),
        ("QPSK Mod", "modulate_mpsk", "bits", {"M": 4}),
        ("8PSK Mod", "modulate_mpsk", "bits", {"M": 8}),
        ("QPSK Demod", "demodulate_mpsk", "signal_from_mod", {"M": 4}),
        ("MFSK Mod", "modulate_mfsk", "bits", {"M": 4}),
        ("MFSK Demod", "demodulate_mfsk", "signal_from_mod", {"M": 4}),
        ("DPSK Mod", "modulate_dpsk", "bits", {}),
        ("DPSK Demod", "demodulate_dpsk", "signal_from_mod", {})
    ]
    results_da = run_category_benchmark("DA", da_tests, N_BITS)
    plot_category_results(results_da, "Digital-to-Analog Benchmark", "benchmark_da.png")

    # --- 3. Analog-to-Digital (AD) Testleri ---
    ad_tests = [
        ("PCM Encode", "encode_pcm", "analog", {"n_bits": 8}),
        ("PCM Decode", "decode_pcm", "signal_from_mod", {"n_bits": 8}),
        ("Delta Encode", "encode_delta_modulation", "analog", {"delta": 0.05}),
        ("Delta Decode", "decode_delta_modulation", "signal_from_mod", {"delta": 0.05})
    ]
    results_ad = run_category_benchmark("AD", ad_tests, N_BITS)
    plot_category_results(results_ad, "Analog-to-Digital Benchmark", "benchmark_ad.png")
    
    print("\n🎉 TÜM BENCHMARK TESTLERİ TAMAMLANDI!")