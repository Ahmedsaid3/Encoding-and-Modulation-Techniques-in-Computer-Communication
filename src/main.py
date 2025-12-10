import streamlit as st
from ui.interface import run_app

if __name__ == "__main__":
    # Sayfa yapılandırması
    st.set_page_config(page_title="Data Comm Simulator", layout="wide")
    
    # Arayüzü başlat
    run_app()