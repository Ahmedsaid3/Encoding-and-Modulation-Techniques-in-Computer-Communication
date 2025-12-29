import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- IMPORT MODULES ---
try:
    from algorithms.dd_encoding import DigitalToDigital
    from algorithms.da_modulation import DigitalToAnalog
    from algorithms.ad_encoding import AnalogToDigital
    from algorithms.aa_modulation import AnalogToAnalog
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.warning("Please make sure to run the project from the root directory using 'streamlit run src/main.py'.")

def run_app():
    # Page Config
    st.set_page_config(page_title="Data Communication Simulator", layout="wide")
    
    # Main Title and Description
    st.title("📡 Data Communication Simulator")
    st.markdown("""
    This project is prepared for the **BLG 337E** course. 
    Select one of the modes below from the sidebar to start the simulation.
    """)
    st.markdown("---")

    # --- SIDEBAR (SETTINGS) ---
    st.sidebar.header("⚙️ Simulation Settings")
    
    # MODE SELECTION
    mode = st.sidebar.radio(
        "Transmission Mode:",
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
                "Select Algorithm:", 
                ["NRZ-L", "NRZI", "Bipolar-AMI", "Pseudoternary", "Manchester", "Differential Manchester"]
            )
            bit_input = st.text_input("Bit Sequence (0 and 1):", "0100110101")
            
        with col2:
            if st.button("Encode & Plot", key="btn_dd"):
                try:
                    bits = [int(b) for b in bit_input if b in '01']
                    if not bits:
                        st.error("Please enter a valid bit sequence!")
                        return

                    dd = DigitalToDigital()
                    
                    # Process based on selection
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
                    
                    # Plotting
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.step(t, s, where='post', linewidth=2, color='blue')
                    ax.set_title(f"{tech} Encoding Signal")
                    ax.set_ylabel("Voltage Level")
                    ax.set_xlabel("Time")
                    ax.grid(True, alpha=0.5)
                    ax.set_ylim(-6, 6)
                    
                    # Draw bit boundaries
                    bit_duration = t[-1] / len(bits)
                    for i in range(len(bits) + 1):
                        ax.axvline(i * bit_duration, color='red', linestyle='--', alpha=0.3)

                    st.pyplot(fig)
                    
                    # Results
                    st.success(f"Original Bits: {bits}")
                    
                    # Clean formatting for decoded bits
                    decoded_clean = [int(b) for b in decoded]
                    st.info(f"Decoded Bits: {decoded_clean}")
                    
                    if list(bits) == decoded_clean:
                        st.markdown("✅ **Success:** Data transmitted without loss.")
                    else:
                        st.markdown("❌ **Error:** Data mismatch.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # --- 2. DIGITAL TO ANALOG (MODULATION) ---
    elif mode == "2. Digital-to-Analog (Modulation)":
        st.header("2️⃣ Digital-to-Analog Modulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            tech = st.selectbox(
                "Modulation Type:", 
                ["ASK", "BFSK", "MFSK (M=4)", "BPSK", "QPSK (M=4)", "8-PSK (M=8)", "DPSK"]
            )
            bit_input = st.text_input("Bit Sequence:", "10110")
            baud = st.slider("Baud Rate (Symbol Rate)", 1, 10, 2)
            fc = st.slider("Carrier Frequency (Hz)", 1, 50, 5)

        with col2:
            if st.button("Modulate", key="btn_da"):
                try:
                    bits = [int(b) for b in bit_input if b in '01']
                    da = DigitalToAnalog()
                    
                    fig, ax = plt.subplots(figsize=(12, 5))
                    
                    recovered = [] # Initialize

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
                        recovered = da.demodulate_dpsk(s, baud_rate=baud, carrier_freq=fc)
                        st.caption("ℹ️ **Note:** Demodulation uses a reference signal with 0 phase to recover the first bit.")

                    # Plotting
                    ax.plot(t, s)
                    ax.set_title(f"{tech} Signal")
                    ax.set_xlabel("Time (s)")
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Demodulation Results with Clean Formatting
                    limit = min(len(bits), len(recovered))
                    
                    # Clean int conversion
                    recovered_clean = [int(b) for b in recovered[:limit]]
                    
                    st.text(f"Original: {bits[:limit]}")
                    st.text(f"Decoded:  {recovered_clean}")

                except Exception as e:
                    st.error(f"Error: {e}")

    # --- 3. ANALOG TO DIGITAL (PCM/DELTA) ---
    elif mode == "3. Analog-to-Digital (Digitization)":
        st.header("3️⃣ Analog-to-Digital Encoding")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            method = st.radio("Method:", ["PCM (Pulse Code Modulation)", "Delta Modulation"])
            freq = st.slider("Analog Signal Frequency (Hz)", 1, 10, 2)
            
            if method == "PCM (Pulse Code Modulation)":
                n_bits = st.slider("Bit Depth (n)", 2, 8, 3)
            else:
                delta = st.slider("Step Size (Delta)", 0.01, 0.5, 0.1)

        with col2:
            if st.button("Digitize", key="btn_ad"):
                ad = AnalogToDigital()
                
                # Analog Signal Generation
                duration = 1.0
                t = np.linspace(0, duration, 200)
                analog_signal = np.sin(2 * np.pi * freq * t)
                
                if method == "PCM (Pulse Code Modulation)":
                    encoded_bits, q_indices = ad.encode_pcm(analog_signal, n_bits=n_bits)
                    reconstructed = ad.decode_pcm(encoded_bits, n_bits=n_bits)
                    
                    st.markdown(f"**PCM Results ({n_bits}-bit):**")
                    st.write(f"Total Bits Generated: {len(encoded_bits)}")
                    
                    # FIX: Clean int conversion for PCM
                    clean_bits = [int(b) for b in encoded_bits[:20]]
                    st.write(f"Bit Stream (First 20): {clean_bits}")
                    
                else: # Delta Modulation
                    encoded_bits, enc_recon = ad.encode_delta_modulation(analog_signal, delta=delta)
                    reconstructed = ad.decode_delta_modulation(encoded_bits, delta=delta)
                    
                    st.markdown(f"**Delta Modulation Results (Δ={delta}):**")
                    
                    # FIX: Clean int conversion for Delta Modulation
                    clean_bits = [int(b) for b in encoded_bits[:20]]
                    st.write(f"Bit Stream (First 20): {clean_bits}")

                # Plotting
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(t, analog_signal, label="Original Analog (Input)", color='blue', alpha=0.4, linewidth=2)
                ax.step(t, reconstructed, where='mid', label="Digitized (Output)", color='red', linewidth=1.5)
                
                ax.set_title(f"{method} Result")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

    # --- 4. ANALOG TO ANALOG (AM/FM/PM) ---
    elif mode == "4. Analog-to-Analog (Modulation)":
        st.header("4️⃣ Analog-to-Analog Modulation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            aa_tech = st.selectbox("Modulation Type:", ["Amplitude Modulation (AM)", "Frequency Modulation (FM)", "Phase Modulation (PM)"])
            
            msg_freq = st.slider("Message Frequency (Hz)", 1, 10, 2)
            carrier_freq = st.slider("Carrier Frequency (Hz)", 20, 100, 50)
            
            # Dynamic Parameters
            param = 0.0
            if aa_tech == "Amplitude Modulation (AM)":
                param = st.slider("Modulation Index (m)", 0.1, 2.0, 0.8)
            elif aa_tech == "Frequency Modulation (FM)":
                param = st.slider("Frequency Sensitivity (kf)", 1.0, 50.0, 20.0)
            elif aa_tech == "Phase Modulation (PM)":
                param = st.slider("Phase Sensitivity (kp)", 0.1, 10.0, 2.0)

        with col2:
            if st.button("Modulate & Demodulate", key="btn_aa"):
                try:
                    fs = 1000 
                    aa = AnalogToAnalog(carrier_freq=carrier_freq, sampling_rate=fs)
                    
                    # Message Signal
                    duration = 1.0
                    t_msg = np.linspace(0, duration, int(duration * fs), endpoint=False)
                    message_signal = np.sin(2 * np.pi * msg_freq * t_msg)
                    
                    # Processing
                    if "AM" in aa_tech:
                        t, mod_signal = aa.modulate_am(message_signal, mod_index=param)
                        demod_signal = aa.demodulate_am(mod_signal)
                    elif "FM" in aa_tech:
                        t, mod_signal = aa.modulate_fm(message_signal, kf=param)
                        demod_signal = aa.demodulate_fm(mod_signal)
                    elif "PM" in aa_tech:
                        t, mod_signal = aa.modulate_pm(message_signal, kp=param)
                        demod_signal = aa.demodulate_pm(mod_signal, kp=param) 

                    # 3 Stacked Plots
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
                    
                    # 1. Message
                    ax1.plot(t_msg, message_signal, 'g')
                    ax1.set_title("1. Message Signal (Original)")
                    ax1.grid(True, alpha=0.3)
                    
                    # 2. Modulated
                    ax2.plot(t, mod_signal, 'b')
                    ax2.set_title(f"2. Modulated Signal ({aa_tech})")
                    ax2.grid(True, alpha=0.3)
                    
                    # 3. Demodulated
                    plot_len = min(len(t), len(demod_signal))
                    ax3.plot(t[:plot_len], demod_signal[:plot_len], 'r')
                    ax3.set_title("3. Demodulated Signal (Recovered)")
                    ax3.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    run_app()