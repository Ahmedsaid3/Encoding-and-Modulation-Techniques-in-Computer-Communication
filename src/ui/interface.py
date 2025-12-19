import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- MODÜLLERİ İMPORT ET ---
# Proje ana dizininden çalıştırıldığı varsayılıyor
try:
    from algorithms.dd_encoding import DigitalToDigital
    from algorithms.da_modulation import DigitalToAnalog
    from algorithms.ad_encoding import AnalogToDigital
    from algorithms.aa_modulation import AnalogToAnalog
except ImportError as e:
    st.error(f"Modül import hatası: {e}")
    st.warning("Lütfen projeyi ana dizinden 'streamlit run src/main.py' komutuyla çalıştırdığınızdan emin olun.")

def run_app():
    st.set_page_config(page_title="Data Communication Simulator", layout="wide")
    
    st.title("📡 Data Communication Simulator")
    st.markdown("""
    Bu proje **BLG 337E** dersi için hazırlanmıştır. 
    Aşağıdaki modlardan birini seçerek simülasyonu başlatabilirsiniz.
    """)
    st.markdown("---")

    # --- KENAR ÇUBUĞU (SIDEBAR) ---
    st.sidebar.header("⚙️ Simülasyon Ayarları")
    
    # MOD SEÇİMİ
    mode = st.sidebar.radio(
        "İletim Modu:",
        [
            "1. Digital-to-Digital (Encoding)",
            "2. Digital-to-Analog (Modulation)",
            "3. Analog-to-Digital (Digitization)",
            "4. Analog-to-Analog (Modulation)"
        ]
    )

    # --- 1. DIGITAL TO DIGITAL (ENCODING) ---
    if mode == "1. Digital-to-Digital (Encoding)":
        st.header("1️⃣ Digital-to-Digital Encoding")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            tech = st.selectbox(
                "Algoritma Seçiniz:", 
                ["NRZ-L", "NRZI", "Bipolar-AMI", "Pseudoternary", "Manchester", "Differential Manchester"]
            )
            bit_input = st.text_input("Bit Dizisi (0 ve 1):", "0100110101")
            
        with col2:
            if st.button("Kodla ve Göster", key="btn_dd"):
                try:
                    bits = [int(b) for b in bit_input if b in '01']
                    if not bits:
                        st.error("Lütfen geçerli bir bit dizisi girin!")
                        return

                    dd = DigitalToDigital()
                    
                    # Seçilen algoritmaya göre işlem yap
                    if tech == "NRZ-L":
                        t, s = dd.encode_nrz_l(bits)
                        decoded = dd.decode_nrz_l(s)
                    elif tech == "NRZI":
                        t, s = dd.encode_nrzi(bits)
                        decoded = dd.decode_nrzi(s)
                    elif tech == "Bipolar-AMI":
                        t, s = dd.encode_bipolar_ami(bits)
                        decoded = dd.decode_bipolar_ami(s)
                    elif tech == "Pseudoternary":
                        t, s = dd.encode_pseudoternary(bits)
                        decoded = dd.decode_pseudoternary(s)
                    elif tech == "Manchester":
                        t, s = dd.encode_manchester(bits)
                        decoded = dd.decode_manchester(s)
                    elif tech == "Differential Manchester":
                        t, s = dd.encode_dif_manch(bits)
                        decoded = dd.decode_dif_manch(s)
                    
                    # Grafik
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.step(t, s, where='post', linewidth=2, color='blue')
                    ax.set_title(f"{tech} Encoding Signal")
                    ax.set_ylabel("Voltaj Seviyesi")
                    ax.set_xlabel("Zaman")
                    ax.grid(True, alpha=0.5)
                    ax.set_ylim(-6, 6)
                    
                    # Bit sınırlarını çiz
                    bit_duration = t[-1] / len(bits)
                    for i in range(len(bits) + 1):
                        ax.axvline(i * bit_duration, color='red', linestyle='--', alpha=0.3)

                    st.pyplot(fig)
                    
                    # Sonuçlar
                    st.success(f"Orijinal Bitler: {bits}")
                    st.info(f"Çözülen Bitler: {list(decoded)}")
                    
                    if list(bits) == list(decoded):
                        st.markdown("✅ **Başarılı:** Veri kayıpsız iletildi.")
                    else:
                        st.markdown("❌ **Hata:** Veri uyuşmazlığı var.")
                        
                except Exception as e:
                    st.error(f"Hata oluştu: {e}")

    # --- 2. DIGITAL TO ANALOG (MODULATION) ---
    elif mode == "2. Digital-to-Analog (Modulation)":
        st.header("2️⃣ Digital-to-Analog Modulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            tech = st.selectbox(
                "Modülasyon Tipi:", 
                ["ASK", "BFSK", "MFSK (M=4)", "BPSK", "QPSK (M=4)", "8-PSK (M=8)", "DPSK"]
            )
            bit_input = st.text_input("Bit Dizisi:", "10110")
            baud = st.slider("Baud Rate (Sembol Hızı)", 1, 10, 2)
            fc = st.slider("Taşıyıcı Frekansı (Hz)", 1, 50, 5)

        with col2:
            if st.button("Modüle Et", key="btn_da"):
                try:
                    bits = [int(b) for b in bit_input if b in '01']
                    da = DigitalToAnalog()
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    
                    if tech == "ASK":
                        t, s = da.modulate_ask(bits, baud_rate=baud, carrier_freq=fc)
                        recovered = da.demodulate_ask(s, baud_rate=baud, carrier_freq=fc)
                        
                    elif tech == "BFSK":
                        t, s = da.modulate_bfsk(bits, baud_rate=baud, freq_0=fc, freq_1=fc*2)
                        recovered = da.demodulate_bfsk(s, baud_rate=baud, freq_0=fc, freq_1=fc*2)
                        st.caption(f"f0: {fc}Hz, f1: {fc*2}Hz")
                        
                    elif tech == "MFSK (M=4)":
                        t, s = da.modulate_mfsk(bits, M=4, baud_rate=baud, base_freq=fc, freq_sep=fc)
                        recovered = da.demodulate_mfsk(s, M=4, baud_rate=baud, base_freq=fc, freq_sep=fc)
                        
                    elif tech == "BPSK":
                        t, s = da.modulate_mpsk(bits, M=2, baud_rate=baud, carrier_freq=fc)
                        recovered = da.demodulate_mpsk(s, M=2, baud_rate=baud, carrier_freq=fc)
                        
                    elif tech == "QPSK (M=4)":
                        t, s = da.modulate_mpsk(bits, M=4, baud_rate=baud, carrier_freq=fc)
                        recovered = da.demodulate_mpsk(s, M=4, baud_rate=baud, carrier_freq=fc)
                        
                    elif tech == "8-PSK (M=8)":
                        t, s = da.modulate_mpsk(bits, M=8, baud_rate=baud, carrier_freq=fc)
                        recovered = da.demodulate_mpsk(s, M=8, baud_rate=baud, carrier_freq=fc)
                        
                    elif tech == "DPSK":
                        t, s = da.modulate_dpsk(bits, baud_rate=baud, carrier_freq=fc)
                        recovered = da.demodulate_dpsk(s, baud_rate=baud)
                        st.info("Not: DPSK demodülasyonunda ilk bit referans eksikliği nedeniyle belirsiz olabilir.")

                    # Çizim
                    ax.plot(t, s)
                    ax.set_title(f"{tech} Sinyali")
                    ax.set_xlabel("Zaman (s)")
                    ax.grid(True, alpha=0.3)
                    
                    # Sembol ayraçları
                    total_duration = t[-1]
                    # MPSK/MFSK için sembol süresi bit süresinden farklı olabilir
                    # Basitlik için grafiği çiziyoruz
                    
                    st.pyplot(fig)
                    
                    # Demodülasyon Sonucu (Uzunluk eşitleyerek göster)
                    limit = min(len(bits), len(recovered))
                    st.text(f"Orijinal: {bits[:limit]}")
                    st.text(f"Çözülen:  {list(recovered[:limit])}")

                except Exception as e:
                    st.error(f"Hata: {e}")

    # --- 3. ANALOG TO DIGITAL (PCM/DELTA) ---
    elif mode == "3. Analog-to-Digital (Digitization)":
        st.header("3️⃣ Analog-to-Digital Encoding")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            method = st.radio("Yöntem:", ["PCM (Pulse Code Modulation)", "Delta Modulation"])
            freq = st.slider("Analog Sinyal Frekansı (Hz)", 1, 10, 2)
            
            if method == "PCM (Pulse Code Modulation)":
                n_bits = st.slider("Bit Derinliği (n)", 2, 8, 3)
            else:
                delta = st.slider("Delta Adımı (Step Size)", 0.01, 0.5, 0.1)

        with col2:
            if st.button("Dönüştür", key="btn_ad"):
                ad = AnalogToDigital()
                
                # Analog Sinyal Üretimi
                duration = 1.0
                t = np.linspace(0, duration, 200) # Görüntü için yüksek çözünürlük
                analog_signal = np.sin(2 * np.pi * freq * t)
                
                if method == "PCM (Pulse Code Modulation)":
                    # PCM Encode / Decode
                    encoded_bits, q_indices = ad.encode_pcm(analog_signal, n_bits=n_bits)
                    reconstructed = ad.decode_pcm(encoded_bits, n_bits=n_bits)
                    
                    st.markdown(f"**PCM Sonuçları ({n_bits}-bit):**")
                    st.write(f"Üretilen Toplam Bit: {len(encoded_bits)}")
                    st.write(f"Bit Akışı (İlk 20): {encoded_bits[:20]}")
                    
                else: # Delta Modulation
                    # DM Encode / Decode
                    encoded_bits, enc_recon = ad.encode_delta_modulation(analog_signal, delta=delta)
                    reconstructed = ad.decode_delta_modulation(encoded_bits, delta=delta)
                    
                    st.markdown(f"**Delta Modulation Sonuçları (Δ={delta}):**")
                    st.write(f"Bit Akışı (İlk 20): {list(encoded_bits[:20])}")

                # Grafikleme
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(t, analog_signal, label="Orijinal Analog (Giriş)", color='blue', alpha=0.4, linewidth=2)
                
                # PCM veya DM çıktısı 'step' grafiğidir
                ax.step(t, reconstructed, where='mid', label="Dijitalleştirilmiş (Çıkış)", color='red', linewidth=1.5)
                
                ax.set_title(f"{method} Sonucu")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    # --- 4. ANALOG TO ANALOG (AM/FM/PM) ---
    elif mode == "4. Analog-to-Analog (Modulation)":
        st.header("4️⃣ Analog-to-Analog Modulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            aa_tech = st.selectbox("Modülasyon Tipi:", ["Amplitude Modulation (AM)", "Frequency Modulation (FM)", "Phase Modulation (PM)"])
            
            msg_freq = st.slider("Mesaj Frekansı (Hz)", 1, 10, 2)
            carrier_freq = st.slider("Taşıyıcı Frekansı (Hz)", 20, 100, 50)
            
            # Dinamik Parametreler
            param = 0.0
            if aa_tech == "Amplitude Modulation (AM)":
                param = st.slider("Modülasyon İndeksi (m)", 0.1, 2.0, 0.8)
            elif aa_tech == "Frequency Modulation (FM)":
                param = st.slider("Frekans Hassasiyeti (kf)", 1.0, 50.0, 20.0)
            elif aa_tech == "Phase Modulation (PM)":
                param = st.slider("Faz Hassasiyeti (kp)", 0.1, 10.0, 2.0)

        with col2:
            if st.button("Modüle Et ve Çöz", key="btn_aa"):
                try:
                    # Örnekleme Hızı Taşıyıcıdan yüksek olmalı
                    fs = 1000 
                    aa = AnalogToAnalog(carrier_freq=carrier_freq, sampling_rate=fs)
                    
                    # Mesaj Sinyali
                    duration = 1.0
                    t_msg = np.linspace(0, duration, int(duration * fs), endpoint=False)
                    message_signal = np.sin(2 * np.pi * msg_freq * t_msg)
                    
                    # İşlemler
                    if "AM" in aa_tech:
                        t, mod_signal = aa.modulate_am(message_signal, mod_index=param)
                        demod_signal = aa.demodulate_am(mod_signal)
                    elif "FM" in aa_tech:
                        t, mod_signal = aa.modulate_fm(message_signal, kf=param)
                        demod_signal = aa.demodulate_fm(mod_signal)
                    elif "PM" in aa_tech:
                        t, mod_signal = aa.modulate_pm(message_signal, kp=param)
                        demod_signal = np.zeros_like(mod_signal) # PM Demod henüz implemente edilmediyse boş döndür
                        st.warning("PM Demodülasyonu bu arayüzde sadece görselleştirme amaçlıdır.")

                    # 3 Alt Alta Grafik
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
                    
                    # 1. Mesaj
                    ax1.plot(t_msg, message_signal, 'g')
                    ax1.set_title("1. Mesaj Sinyali (Orijinal)")
                    ax1.grid(True, alpha=0.3)
                    
                    # 2. Modüleli
                    ax2.plot(t, mod_signal, 'b')
                    ax2.set_title(f"2. Modüle Edilmiş Sinyal ({aa_tech})")
                    ax2.grid(True, alpha=0.3)
                    
                    # 3. Demodüleli
                    if "PM" not in aa_tech:
                        # Boyut eşitleme (Convolution/Diff nedeniyle kayma olabilir)
                        plot_len = min(len(t), len(demod_signal))
                        ax3.plot(t[:plot_len], demod_signal[:plot_len], 'r')
                        ax3.set_title("3. Demodüle Edilmiş Sinyal (Kurtarılan)")
                        ax3.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Hata: {e}")

if __name__ == "__main__":
    run_app()