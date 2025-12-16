import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Kendi yazdığımız modülleri import ediyoruz
# Not: Python'ın bu modülleri bulabilmesi için projeyi ana dizinden calistirmalisin
from algorithms.dd_encoding import DigitalToDigital
from algorithms.da_modulation import DigitalToAnalog
from algorithms.ad_encoding import AnalogToDigital
# from algorithms.aa_modulation import AnalogToAnalog (Henüz yazmadık)

def run_app():
    st.title("Data Communication Simulator")
    st.markdown("Bu proje **BLG 337E** dersi için hazırlanmıştır.")

    # --- KENAR ÇUBUĞU (SIDEBAR) ---
    st.sidebar.header("Ayarlar")
    
    # 1. Mod Seçimi
    mode = st.sidebar.selectbox(
        "İletim Modunu Seçiniz:",
        [
            "Digital-to-Digital (Encoding)",
            "Digital-to-Analog (Modulation)",
            "Analog-to-Digital (PCM)",
            "Analog-to-Analog (AM/FM)"
        ]
    )

    # --- ANA EKRAN ---
    
    # 1. DIGITAL TO DIGITAL EKRANI
    if mode == "Digital-to-Digital (Encoding)":
        st.header("Digital-to-Digital Encoding")
        
        # Kullanıcıdan Girdi Al
        tech = st.selectbox("Algoritma Seç:", ["NRZ-L", "Bipolar-AMI", "Manchester"])
        bit_input = st.text_input("Bit Dizisi Girin (Örn: 010011):", "010011")
        
        if st.button("Simüle Et"):
            # String girdiyi listeye çevir: "010" -> [0, 1, 0]
            try:
                bits = [int(b) for b in bit_input if b in '01']
                
                # Sınıfı çağır
                dd = DigitalToDigital()
                
                if tech == "NRZ-L":
                    t, s = dd.encode_nrz_l(bits)
                    decoded = dd.decode_nrz_l(s)
                elif tech == "Bipolar-AMI":
                    t, s = dd.encode_bipolar_ami(bits)
                    decoded = dd.decode_bipolar_ami(s)
                elif tech == "Manchester":
                    t, s = dd.encode_manchester(bits)
                    decoded = dd.decode_manchester(s)
                
                # Grafiği Çiz
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.step(t, s, where='post', linewidth=2)
                ax.set_title(f"{tech} Encoding")
                ax.grid(True)
                ax.set_ylim(-6, 6)
                
                # Streamlit'e grafiği bas
                st.pyplot(fig)
                
                # Sonuçları Yazdır
                st.success(f"Gönderilen: {bits}")
                st.info(f"Çözülen:    {list(decoded)}")
                
            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")

    # 2. DIGITAL TO ANALOG EKRANI
    elif mode == "Digital-to-Analog (Modulation)":
        st.header("Digital-to-Analog Modulation")
        
        tech = st.selectbox("Algoritma Seç:", ["ASK", "FSK", "PSK (BPSK/QPSK)"])
        bit_input = st.text_input("Bit Dizisi Girin:", "10110")
        
        # Parametreler (Slider ile)
        col1, col2 = st.columns(2)
        with col1:
            baud = st.slider("Baud Rate", 1, 10, 1)
        with col2:
            freq = st.slider("Carrier Frequency (Hz)", 1, 20, 5)

        if st.button("Modüle Et"):
            bits = [int(b) for b in bit_input if b in '01']
            da = DigitalToAnalog()
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            if tech == "ASK":
                t, s = da.modulate_ask(bits, baud_rate=baud, carrier_freq=freq)
                ax.plot(t, s)
            elif tech == "FSK":
                # FSK için ikinci frekans lazım
                freq2 = freq * 2
                t, s = da.modulate_fsk(bits, baud_rate=baud, freq_0=freq, freq_1=freq2)
                ax.plot(t, s)
                st.caption(f"Freq 0: {freq}Hz, Freq 1: {freq2}Hz")
            elif tech == "PSK (BPSK/QPSK)":
                # Şimdilik BPSK (M=2) yapalım
                t, s = da.modulate_mpsk(bits, M=2, baud_rate=baud, carrier_freq=freq)
                ax.plot(t, s)
            
            ax.grid(True)
            st.pyplot(fig)

    # 3. ANALOG TO DIGITAL EKRANI
    elif mode == "Analog-to-Digital (PCM)":
        st.header("Analog-to-Digital (PCM)")
        st.write("Analog bir sinüs dalgasını dijitalleştirelim.")
        
        n_bits = st.slider("Bit Derinliği (Resolution)", 1, 8, 3)
        
        if st.button("Dönüştür"):
            ad = AnalogToDigital()
            
            # Örnek Sinyal Oluştur
            t = np.linspace(0, 1, 100) # Analog gibi görünen yüksek çözünürlüklü
            analog_signal = np.sin(2 * np.pi * 2 * t) # 2 Hz sinüs
            
            # Encode
            encoded_bits, q_indices = ad.encode_pcm(analog_signal, n_bits=n_bits)
            # Decode (Reconstruct)
            reconstructed = ad.decode_pcm(encoded_bits, n_bits=n_bits)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t, analog_signal, label="Analog Input", alpha=0.5)
            ax.step(t, reconstructed, where='mid', label="PCM Output", color='red')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            st.text(f"Oluşan Bit Dizisi (İlk 20): {encoded_bits[:20]}...")

    # 4. ANALOG TO ANALOG (EKSİK KISIM)
    elif mode == "Analog-to-Analog (AM/FM)":
        st.warning("⚠️ Bu modül henüz kodlanmadı! (AM/FM)")
        st.info("Sıradaki adımda burayı kodlayabiliriz.")