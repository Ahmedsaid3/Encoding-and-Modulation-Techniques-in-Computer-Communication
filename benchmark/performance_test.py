import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# --- PATH SETTINGS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- IMPORTS ---
try:
    # 1. ORIGINAL (Reference)
    from src.algorithms.dd_encoding import DigitalToDigital as OrigDD
    from src.algorithms.da_modulation import DigitalToAnalog as OrigDA
    from src.algorithms.ad_encoding import AnalogToDigital as OrigAD
    from src.algorithms.aa_modulation import AnalogToAnalog as OrigAA
    
    # 2. V1 (ChatGPT)
    try:
        from ai_optimized_v1.algorithms.dd_encoding import DigitalToDigital as ChatGptDD
        from ai_optimized_v1.algorithms.da_modulation import DigitalToAnalog as ChatGptDA
        from ai_optimized_v1.algorithms.ad_encoding import AnalogToDigital as ChatGptAD
        from ai_optimized_v1.algorithms.aa_modulation import AnalogToAnalog as ChatGptAA
    except ImportError:
        print("⚠️ Warning: Some V1 (ChatGPT) modules are missing. Using Original for missing ones.")
        from src.algorithms.dd_encoding import DigitalToDigital as ChatGptDD
        from src.algorithms.da_modulation import DigitalToAnalog as ChatGptDA
        from src.algorithms.ad_encoding import AnalogToDigital as ChatGptAD
        from src.algorithms.aa_modulation import AnalogToAnalog as ChatGptAA

    # 3. V2 (GEMINI)
    from ai_optimized_v2.algorithms.dd_encoding import DigitalToDigital as GeminiDD
    from ai_optimized_v2.algorithms.da_modulation import DigitalToAnalog as GeminiDA
    from ai_optimized_v2.algorithms.ad_encoding import AnalogToDigital as GeminiAD
    from ai_optimized_v2.algorithms.aa_modulation import AnalogToAnalog as GeminiAA

    print("✅ All Modules (Original, ChatGPT, Gemini) loaded successfully.")

except ImportError as e:
    print(f"❌ Module Error: {e}")
    print("⚠️ Please ensure 'src', 'ai_optimized_v1', and 'ai_optimized_v2' directories exist.")
    sys.exit(1)

def run_category_benchmark(category_name, tests, n_bits):
    """
    Runs tests for a specific category (DD, DA, AD, or AA) and returns results.
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK CATEGORY: {category_name} (Data Size: {n_bits} points)")
    print(f"{'='*70}")

    # Prepare Data
    bits = np.random.randint(0, 2, n_bits)
    
    # Analog signal setup
    duration = 1.0
    fs = n_bits 
    t = np.linspace(0, duration, fs)
    # Complex analog signal for testing
    analog_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    
    # Initialize Instances
    instances = {
        "DD": {
            "Orig": OrigDD(), "ChatGPT": ChatGptDD(), "Gemini": GeminiDD()
        },
        "DA": {
            "Orig": OrigDA(), "ChatGPT": ChatGptDA(), "Gemini": GeminiDA()
        },
        "AD": {
            "Orig": OrigAD(), "ChatGPT": ChatGptAD(), "Gemini": GeminiAD()
        },
        "AA": {
            "Orig": OrigAA(carrier_freq=100, sampling_rate=fs), 
            "ChatGPT": ChatGptAA(carrier_freq=100, sampling_rate=fs), 
            "Gemini": GeminiAA(carrier_freq=100, sampling_rate=fs)
        }
    }
    
    current_instances = instances[category_name]
    results = {}

    for test_label, method_name, input_type, kwargs in tests:
        print(f"--- Test: {test_label} ---")
        
        # Determine Arguments
        if input_type == "bits":
            args = [bits]
        elif input_type == "analog":
            args = [analog_signal]
        elif input_type == "signal_from_mod": 
            # Need a modulated signal for demodulation test
            temp_mod_method = method_name.replace("demodulate", "modulate").replace("decode", "encode")
            if "pcm" in method_name: temp_mod_method = "encode_pcm"
            
            # Generate signal using Original (Fair start)
            if category_name == "AA":
                signal_arg = getattr(current_instances["Orig"], temp_mod_method)(analog_signal, **kwargs)[1]
            elif "pcm" in method_name:
                signal_arg = current_instances["Orig"].encode_pcm(analog_signal, **kwargs)[0] 
            else:
                signal_arg = getattr(current_instances["Orig"], temp_mod_method)(bits, **kwargs)[1]
            args = [signal_arg]

        times = []
        # Run for each version
        for version_name in ["Orig", "ChatGPT", "Gemini"]:
            obj = current_instances[version_name]
            
            try:
                start = time.perf_counter()
                getattr(obj, method_name)(*args, **kwargs)
                duration = time.perf_counter() - start
            except Exception as e:
                print(f"  ⚠️ {version_name} Error: {e}")
                duration = 0 
            
            times.append(duration)

        t_orig, t_v1, t_v2 = times
        
        # Calculate Speedups
        s_v1 = t_orig / t_v1 if t_v1 > 1e-9 else 0
        s_v2 = t_orig / t_v2 if t_v2 > 1e-9 else 0
        
        print(f"  ⏱️  Orig: {t_orig:.4f}s | ChatGPT: {t_v1:.4f}s | Gemini: {t_v2:.4f}s")
        print(f"  🚀 ChatGPT Speedup: {s_v1:.1f}x | Gemini Speedup: {s_v2:.1f}x")
        
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
    
    # Bars
    rects1 = ax.bar(x - width, t_orig, width, label='Original', color='#ff7675')
    rects2 = ax.bar(x,        t_v1,   width, label='ChatGPT (V1)', color='#74b9ff')
    rects3 = ax.bar(x + width, t_v2,   width, label='Gemini (V2)', color='#55efc4')

    # Labels and Title
    ax.set_ylabel('Execution Time (Seconds) - Lower is Better')
    ax.set_title(f'{category_name} Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Y eksenini biraz genişletelim ki tepedeki yazılar kesilmesin
    # En yüksek barın %15 fazlası kadar yer açıyoruz
    max_height = max(max(t_orig), max(t_v1), max(t_v2))
    ax.set_ylim(0, max_height * 1.2)

    # --- Speedup Labels for ChatGPT (V1) ---
    for i in range(len(labels)):
        if t_v1[i] > 1e-9:
            speedup = t_orig[i] / t_v1[i]
            # Yazıyı dik (90 derece) yazdırıyoruz ve biraz yukarı (xytext) kaydırıyoruz
            ax.annotate(f'{speedup:.1f}x',
                        xy=(x[i], t_v1[i]),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90,
                        fontsize=12, fontweight='bold', color='#2980b9')

    # --- Speedup Labels for Gemini (V2) ---
    for i in range(len(labels)):
        if t_v2[i] > 1e-9:
            speedup = t_orig[i] / t_v2[i]
            # Yazıyı dik (90 derece) yazdırıyoruz
            ax.annotate(f'{speedup:.1f}x',
                        xy=(x[i] + width, t_v2[i]),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90,
                        fontsize=12, fontweight='bold', color='#006266')

    plt.tight_layout()
    save_path = f'benchmark/{filename}'
    plt.savefig(save_path)
    print(f"✅ Plot saved: {save_path}")

if __name__ == "__main__":
    # Data Size for Load Testing
    N_BITS = 50000 

# --- 1. Digital-to-Digital (DD) Tests ---
    dd_tests = [
        # NRZ-L
        ("NRZ-L Enc", "encode_nrz_l", "bits", {}),
        ("NRZ-L Dec", "decode_nrz_l", "signal_from_mod", {}),
        
        # Bipolar AMI
        ("AMI Enc", "encode_bipolar_ami", "bits", {}),
        ("AMI Dec", "decode_bipolar_ami", "signal_from_mod", {}),
        
        # Manchester
        ("Manch. Enc", "encode_manchester", "bits", {}),
        ("Manch. Dec", "decode_manchester", "signal_from_mod", {}),
        
        # NRZI
        ("NRZI Enc", "encode_nrzi", "bits", {}),
        ("NRZI Dec", "decode_nrzi", "signal_from_mod", {}),
        
        # Pseudoternary
        ("Pseudoter. Enc", "encode_pseudoternary", "bits", {}),
        ("Pseudoter. Dec", "decode_pseudoternary", "signal_from_mod", {}),
        
        # Differential Manchester
        ("Diff. Manch. Enc", "encode_dif_manch", "bits", {}),
        ("Diff. Manch. Dec", "decode_dif_manch", "signal_from_mod", {})
    ]
    results_dd = run_category_benchmark("DD", dd_tests, N_BITS)
    plot_category_results(results_dd, "Digital-to-Digital Benchmark", "benchmark_dd.png")

    # --- 2. Digital-to-Analog (DA) Tests ---
    da_tests = [
        ("ASK Mod", "modulate_ask", "bits", {}),
        ("ASK Demod", "demodulate_ask", "signal_from_mod", {}),
        ("BFSK Mod", "modulate_bfsk", "bits", {}),
        ("BFSK Demod", "demodulate_bfsk", "signal_from_mod", {}),
        ("BPSK Mod", "modulate_mpsk", "bits", {"M": 2}),
        ("BPSK Demod", "demodulate_mpsk", "signal_from_mod", {"M": 2}),
        ("QPSK Mod", "modulate_mpsk", "bits", {"M": 4}),
        ("QPSK Demod", "demodulate_mpsk", "signal_from_mod", {"M": 4}),
        ("8PSK Mod", "modulate_mpsk", "bits", {"M": 8}),
        ("8PSK Demod", "demodulate_mpsk", "signal_from_mod", {"M": 8}),
        ("MFSK Mod", "modulate_mfsk", "bits", {"M": 4}),
        ("MFSK Demod", "demodulate_mfsk", "signal_from_mod", {"M": 4}),
        ("DPSK Mod", "modulate_dpsk", "bits", {}),
        ("DPSK Demod", "demodulate_dpsk", "signal_from_mod", {})
    ]
    results_da = run_category_benchmark("DA", da_tests, N_BITS)
    plot_category_results(results_da, "Digital-to-Analog Benchmark", "benchmark_da.png")

    # --- 3. Analog-to-Digital (AD) Tests ---
    ad_tests = [
        ("PCM Encode", "encode_pcm", "analog", {"n_bits": 8}),
        ("PCM Decode", "decode_pcm", "signal_from_mod", {"n_bits": 8}),
        ("Delta Encode", "encode_delta_modulation", "analog", {"delta": 0.05}),
        ("Delta Decode", "decode_delta_modulation", "signal_from_mod", {"delta": 0.05})
    ]
    results_ad = run_category_benchmark("AD", ad_tests, N_BITS)
    plot_category_results(results_ad, "Analog-to-Digital Benchmark", "benchmark_ad.png")

    # --- 4. Analog-to-Analog (AA) Tests ---
    aa_tests = [
        ("AM Mod", "modulate_am", "analog", {}),
        ("AM Demod", "demodulate_am", "signal_from_mod", {}),
        ("FM Mod", "modulate_fm", "analog", {}),
        ("FM Demod", "demodulate_fm", "signal_from_mod", {}),
        ("PM Mod", "modulate_pm", "analog", {"kp": 2.0}),
        ("PM Demod", "demodulate_pm", "signal_from_mod", {"kp": 2.0})
    ]
    results_aa = run_category_benchmark("AA", aa_tests, 50000)
    plot_category_results(results_aa, "Analog-to-Analog Benchmark", "benchmark_aa.png")
    
    print("\n🎉 ALL BENCHMARK TESTS COMPLETED!")