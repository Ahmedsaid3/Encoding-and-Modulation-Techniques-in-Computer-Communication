# 📡 Data Communication Simulator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![NumPy](https://img.shields.io/badge/NumPy-Vectorized-013243)

**BLG 337E - Principles of Computer Communication Project**

This project is a comprehensive simulation tool designed to visualize and analyze fundamental data communication techniques. It implements Digital-to-Digital, Digital-to-Analog, Analog-to-Digital, and Analog-to-Analog conversion algorithms. 

Ideally suited for educational purposes, it allows users to modify parameters (frequency, baud rate, bit depth) and observe the resulting waveforms in real-time.

---

## 🚀 Features

The simulator covers four main domains of data transmission:

### 1. Digital-to-Digital Encoding (Baseband)
* **NRZ-L** (Non-Return-to-Zero Level)
* **NRZI** (Non-Return-to-Zero Inverted)
* **Bipolar-AMI** (Alternate Mark Inversion)
* **Pseudoternary**
* **Manchester**
* **Differential Manchester**

### 2. Digital-to-Analog Modulation (Broadband)
* **ASK** (Amplitude Shift Keying)
* **BFSK** (Binary Frequency Shift Keying)
* **MFSK** (Multiple Frequency Shift Keying)
* **BPSK** (Binary Phase Shift Keying)
* **QPSK** (Quadrature Phase Shift Keying)
* **8-PSK** (8-Phase Shift Keying)
* **DPSK** (Differential Phase Shift Keying)

### 3. Analog-to-Digital Conversion
* **PCM** (Pulse Code Modulation) - *Includes Sampling & Quantization*
* **Delta Modulation** (DM)

### 4. Analog-to-Analog Modulation
* **AM** (Amplitude Modulation)
* **FM** (Frequency Modulation)
* **PM** (Phase Modulation)

---

## 🧠 AI-Based Optimization

One of the key requirements of this project was to optimize algorithm performance using AI tools. 
The codebase includes two implementations:
1.  **Iterative (Original):** Standard Python loops (Baseline).
2.  **Vectorized (AI-Optimized):** Optimized using NumPy vectorization and broadcasting techniques (suggested by Gemini & ChatGPT).

**Performance Result:** The vectorized approach achieved speedups ranging from **20x to 100x** for complex modulation schemes like QPSK and PCM.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/data-comm-simulator.git](https://github.com/yourusername/data-comm-simulator.git)
    cd data-comm-simulator
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Mac/Linux:
    source venv/bin/activate
    # Windows:
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is missing, install manually: `pip install numpy matplotlib scipy streamlit`)*

---

## 🖥️ Usage

### Running the User Interface
To start the interactive simulator:

```bash
streamlit run src/ui/interface.py

```

*Depending on your folder structure, you might need to run `streamlit run interface.py` if it is in the root directory.*

This will open a web browser where you can:

1. Select the transmission mode.
2. Enter custom bits or configure analog signals.
3. View the generated waveforms and decoded outputs.

### Running Performance Benchmarks

To compare the speed of Original vs. AI-Optimized algorithms and generate performance plots:

```bash
python benchmark/performance_test.py

```

This script will:

* Run load tests on all algorithms.
* Compare execution times.
* Generate charts in the `benchmark/` folder (e.g., `benchmark_da.png`).

---

## 📂 Project Structure

```
data-comm-simulator/
├── src/
│   ├── algorithms/          # Core logic (DD, DA, AD, AA classes)
│   │   ├── dd_encoding.py
│   │   ├── da_modulation.py
│   │   ├── ad_encoding.py
│   │   └── aa_modulation.py
│   └── ui/
│       └── interface.py     # Streamlit UI code
├── ai_optimized_v2/         # Gemini Optimized Algorithms
├── benchmark/               # Performance testing scripts & results
├── requirements.txt         # Dependencies
└── README.md                # This file

```

---

## 👥 Authors

* **Furkan Yalçın**
* **Ahmed Said Gülşen**

Istanbul Technical University

Faculty of Computer and Informatics

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

