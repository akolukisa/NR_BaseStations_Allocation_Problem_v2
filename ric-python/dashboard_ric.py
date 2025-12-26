#!/usr/bin/env python3
"""
Interactive RIC Beam Assignment Dashboard
Modern Web-based UI with Real-time Visualization
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import subprocess
import shutil
from pathlib import Path
from matplotlib.patches import Circle, Wedge
import plotly.graph_objects as go
import plotly.express as px
import time
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster

# ============================================================
# NS-3/5G-LENA CONFIGURATION
# ============================================================
NS3_DIR = '/Users/akolukisa/Downloads/ns-3.46'
PYTHON_PATH = '/opt/homebrew/bin/python3.11'
LENA_OUTPUT_DIR = Path('/Users/akolukisa/FinalThesis/ric-python')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="5G RIC Beam Assignment Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Modern Dark Theme
st.markdown("""
    <style>
    /* Modern dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .main {
        padding-top: 1rem;
    }
    
    /* Sidebar - Modern glassmorphism style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p {
        color: #cbd5e1 !important;
    }
    
    /* Selectbox modern style */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
    }
    
    /* Multiselect tags - Cyan/Teal gradient */
    span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }
    span[data-baseweb="tag"] span {
        color: white !important;
    }
    
    /* Slider modern style */
    .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.9rem;
    }
    .stSlider [data-baseweb="slider"] {
        padding: 8px 0;
    }
    
    /* Checkbox modern */
    .stCheckbox label {
        color: #cbd5e1 !important;
    }
    
    /* Primary Button - Modern gradient */
    button[kind="primary"] {
        background: linear-gradient(135deg, #06b6d4 0%, #0284c7 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4) !important;
    }
    
    /* Secondary Button */
    button[kind="secondary"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.3) !important;
        border-radius: 10px !important;
        color: #cbd5e1 !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(148, 163, 184, 0.2);
        margin: 1.5rem 0;
    }
    
    /* Section subheaders with icon */
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem;
        letter-spacing: 0.5px;
        margin-top: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    /* Info/Help icons */
    [data-testid="stSidebar"] .stTooltipIcon {
        color: #64748b;
    }
    
    /* Main content headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_existing_lena_simulations(beam_count: int):
    """
    Belirli beam sayısı için mevcut LENA simülasyon klasörlerini bul
    Returns: [(folder_name, folder_path, creation_date, ue_counts), ...]
    """
    simulations = []
    
    # Eski format klasörler (lena_Xbeam_10to100)
    old_format = LENA_OUTPUT_DIR / f'lena_{beam_count}beam_10to100'
    if old_format.exists():
        # Klasörün oluşturulma tarihini al
        mtime = old_format.stat().st_mtime
        date_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        # Mevcut UE klasörlerini say
        ue_dirs = [d.name for d in old_format.iterdir() if d.is_dir() and d.name.startswith('ue')]
        ue_counts = sorted([int(d.replace('ue', '')) for d in ue_dirs if d.replace('ue', '').isdigit()])
        simulations.append({
            'name': f'lena_{beam_count}beam_10to100',
            'path': str(old_format),
            'date': date_str,
            'ue_counts': ue_counts,
            'display': f"{date_str} - {len(ue_counts)} UE ({min(ue_counts) if ue_counts else 0}-{max(ue_counts) if ue_counts else 0})"
        })
    
    # Yeni format klasörler (lena_Xbeam_YYYYMMDD_HHMMSS)
    for folder in LENA_OUTPUT_DIR.glob(f'lena_{beam_count}beam_20*'):
        if folder.is_dir():
            # Tarih/saat bilgisini klasör adından çıkar
            parts = folder.name.split('_')
            if len(parts) >= 3:
                try:
                    date_part = parts[2]
                    time_part = parts[3] if len(parts) > 3 else '000000'
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}"
                except:
                    date_str = "Bilinmiyor"
            else:
                date_str = "Bilinmiyor"
            
            # Mevcut UE klasörlerini say
            ue_dirs = [d.name for d in folder.iterdir() if d.is_dir() and d.name.startswith('ue')]
            ue_counts = sorted([int(d.replace('ue', '')) for d in ue_dirs if d.replace('ue', '').isdigit()])
            
            simulations.append({
                'name': folder.name,
                'path': str(folder),
                'date': date_str,
                'ue_counts': ue_counts,
                'display': f"{date_str} - {len(ue_counts)} UE ({min(ue_counts) if ue_counts else 0}-{max(ue_counts) if ue_counts else 0})"
            })
    
    # Tarihe göre sırala (en yeni önce)
    simulations.sort(key=lambda x: x['date'], reverse=True)
    return simulations


def create_timestamped_folder(beam_count: int):
    """
    Yeni LENA simülasyonu için tarih/saatli klasör oluştur
    """
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    folder_name = f'lena_{beam_count}beam_{timestamp}'
    folder_path = LENA_OUTPUT_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_name, str(folder_path)


def run_lena_simulation(num_ues: int, num_beams: int, num_gnbs: int, output_subdir: str, 
                        tx_power: int = 46, bandwidth: int = 100, numerology: int = 1,
                        sim_time: int = 500, frequency: float = 3.5) -> tuple:
    """
    Gerçek ns-3/5G-LENA simülasyonunu çalıştır
    
    Parameters:
        num_ues: Toplam UE sayısı
        num_beams: Hüzme sayısı
        num_gnbs: gNB sayısı
        output_subdir: Çıktı alt klasörü
        tx_power: gNB iletim gücü (dBm)
        bandwidth: Bant genişliği (MHz)
        numerology: NR numerology (0, 1, 2)
        sim_time: Simülasyon süresi (ms)
        frequency: Merkez frekans (GHz)
    
    Returns:
        (success: bool, actual_ues: int, message: str)
    """
    # UE sayısını gNB sayısına böl (en az hedef UE sayısı olacak şekilde yukarı yuvarla)
    import math
    ue_per_gnb = max(1, math.ceil(num_ues / num_gnbs))
    actual_total_ues = ue_per_gnb * num_gnbs
    output_path = LENA_OUTPUT_DIR / output_subdir / f'ue{num_ues}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Eski SINR dosyalarını temizle VE silindiğini doğrula
    thesis_results = Path(NS3_DIR) / 'thesis-results'
    if thesis_results.exists():
        for f in thesis_results.glob('sinr_*.json'):
            try:
                f.unlink()
            except PermissionError:
                # Dosya kilitli olabilir, zorla sil
                import os
                os.system(f'rm -f "{f}"')
            except Exception:
                pass
        
        # Doğrula: tüm sinr dosyaları silindi mi?
        remaining = list(thesis_results.glob('sinr_*.json'))
        if remaining:
            # Zorla sil
            import os
            os.system(f'rm -f {thesis_results}/sinr_*.json')
            import time as time_module
            time_module.sleep(0.5)  # Filesystem sync bekle
    
    # ns-3 komutunu oluştur (tüm parametrelerle)
    bandwidth_hz = bandwidth * 1e6  # MHz -> Hz
    frequency_hz = frequency * 1e9  # GHz -> Hz
    
    cmd = (f'cd {NS3_DIR} && {PYTHON_PATH} ns3 run "thesis-nr-scenario '
           f'--logging=false '
           f'--gNbNum={num_gnbs} '
           f'--ueNumPergNb={ue_per_gnb} '
           f'--numBeams={num_beams} '
           f'--totalTxPower={tx_power} '
           f'--bandwidth={bandwidth_hz:.0f} '
           f'--numerology={numerology} '
           f'--centralFrequency={frequency_hz:.0f} '
           f'--simTime={sim_time}ms" 2>&1')
    
    try:
        # ns-3 çalıştırılıyor - bu gerçekten uzun sürmeli!
        import time as time_module
        start_time = time_module.time()
        
        # Simülasyon öncesi dosya sayısını kontrol et
        pre_files = list(thesis_results.glob('sinr_gnb*_*.json'))
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        elapsed = time_module.time() - start_time
        
        # Filesystem sync bekle
        time_module.sleep(0.5)
        
        # Simülasyon çok hızlı bittiyse uyarı ver (< 5 saniye)
        if elapsed < 5.0 and num_ues > 10:
            print(f"\n⚠️ UYARI: {num_ues} UE simülasyonu sadece {elapsed:.1f}s sürdü! ns-3 çalışmamış olabilir.\n")
        
        # SINR dosyalarını kontrol et ve kopyala
        sinr_files = list(thesis_results.glob('sinr_gnb*_*.json'))
        
        if len(sinr_files) >= num_gnbs:
            # Dosyaları hedef klasöre kopyala
            for f in sinr_files:
                shutil.copy(f, output_path / f.name)
            
            # Doğrulama: Dosyalardaki UE sayısını kontrol et
            actual_ues = 0
            for f in sinr_files:
                try:
                    with open(f, 'r') as fh:
                        data = json.load(fh)
                        actual_ues += data.get('num_ues', 0)
                except:
                    pass
            
            expected_ues = ue_per_gnb * num_gnbs
            if abs(actual_ues - expected_ues) > num_gnbs:  # Tolerans
                print(f"\n⚠️ UYARI: Beklenen {expected_ues} UE, dosyalarda {actual_ues} UE bulundu!\n")
            
            return True, actual_ues, f"✅ {num_ues} UE tamamlandı ({elapsed:.1f}s, gerçek:{actual_ues})"
        else:
            # ns-3 çıktısını kontrol et
            error_msg = result.stderr[:200] if result.stderr else result.stdout[:200]
            return False, 0, f"⚠️ SINR dosyası yok ({elapsed:.1f}s): {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, 0, "⏱️ Simülasyon zaman aşımına uğradı (180s)"
    except Exception as e:
        return False, 0, f"❌ Hata: {str(e)}"


def regenerate_benchmark_csv(output_subdir: str, num_beams: int, ue_counts: list = None):
    """
    LENA simülasyonları tamamlandıktan sonra benchmark CSV dosyası oluştur
    """
    import csv
    
    # Seçili UE sayılarını kullan (yoksa default)
    if ue_counts is None:
        ue_targets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    else:
        ue_targets = sorted(ue_counts)
    
    algorithms = ['Max-SINR', 'GA', 'HGA', 'PBIG']
    results = []
    
    for ue_count in ue_targets:
        sinr_matrix, actual_ues = load_lena_sinr(output_subdir, ue_count)
        if sinr_matrix is None:
            continue
        
        for alg_name in algorithms:
            if alg_name == 'Max-SINR':
                assignment = max_sinr_assignment(sinr_matrix)
            elif alg_name == 'GA':
                assignment = ga_algorithm(sinr_matrix, alpha=1.0)
            elif alg_name == 'HGA':
                assignment = hga_algorithm(sinr_matrix, alpha=1.0)
            else:  # PBIG
                assignment = pbig_algorithm(sinr_matrix, alpha=1.0)
            
            rates = compute_rates(sinr_matrix, assignment, interference_factor=0.5)
            sum_rate = np.sum(rates)
            throughput = sum_rate * 100 * 0.8
            fairness = jain_fairness(rates)
            
            results.append({
                'algorithm': alg_name,
                'num_ues': ue_count,
                'sum_rate': sum_rate,
                'throughput': throughput,
                'fairness': fairness
            })
    
    # CSV'ye yaz
    csv_filename = f'results_{num_beams}beam_10to100.csv'
    csv_path = LENA_OUTPUT_DIR / csv_filename
    
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['algorithm', 'num_ues', 'sum_rate', 'throughput', 'fairness'])
            writer.writeheader()
            writer.writerows(results)
        return True, csv_filename
    return False, None

def load_lena_sinr(base_dir, ue_count):
    """Load real 5G-LENA SINR data"""
    # Tam path oluştur
    if os.path.isabs(base_dir):
        ue_dir = os.path.join(base_dir, f'ue{ue_count}')
    else:
        ue_dir = os.path.join(str(LENA_OUTPUT_DIR), base_dir, f'ue{ue_count}')
    
    if not os.path.exists(ue_dir):
        return None, 0
    
    all_sinr = []
    all_ues = []
    
    for gnb_id in range(3):
        for time_ms in ['200ms', '300ms', '400ms']:  # 200ms önce kontrol et
            filepath = os.path.join(ue_dir, f'sinr_gnb{gnb_id}_{time_ms}.json')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                sinr = np.array(data['sinr_matrix_dB'])
                all_sinr.append(sinr)
                all_ues.extend(data['ue_ids'])
                break
    
    if not all_sinr:
        return None, 0
    
    # Pad matrices to have same number of UEs, then stack vertically
    # Find max UE count across all gNBs
    max_ues = max(m.shape[1] for m in all_sinr)
    
    padded_sinr = []
    for i, sinr in enumerate(all_sinr):
        if sinr.shape[1] < max_ues:
            # Pad with very low SINR (-100 dB) for non-existent UEs
            padding = np.full((sinr.shape[0], max_ues - sinr.shape[1]), -100.0)
            sinr_padded = np.hstack([sinr, padding])
        else:
            sinr_padded = sinr
        padded_sinr.append(sinr_padded)
    
    # Stack vertically: 3 gNBs × num_beams = total beams
    try:
        combined = np.vstack(padded_sinr)
    except ValueError as e:
        # Debug: Print shapes if vstack fails
        print(f"Error stacking matrices for {ue_count} UEs:")
        for i, m in enumerate(padded_sinr):
            print(f"  gNB {i}: shape {m.shape}")
        raise
    
    return combined, len(set(all_ues))

def compute_rates(sinr_matrix, assignment, interference_factor=0.5):
    """Compute rates with interference penalty"""
    num_beams, num_ues = sinr_matrix.shape
    rates = np.zeros(num_ues)
    
    beam_loads = {}
    for ue, beam in enumerate(assignment):
        if beam not in beam_loads:
            beam_loads[beam] = []
        beam_loads[beam].append(ue)
    
    for ue, beam in enumerate(assignment):
        original_sinr_db = sinr_matrix[beam, ue]
        num_sharing = len(beam_loads.get(beam, [1]))
        
        if num_sharing > 1:
            interference_penalty = interference_factor * (num_sharing - 1)
            effective_sinr_db = original_sinr_db - interference_penalty
        else:
            effective_sinr_db = original_sinr_db
        
        sinr_linear = 10 ** (effective_sinr_db / 10)
        rate = np.log2(1 + max(0.001, sinr_linear))
        rates[ue] = max(0.01, rate)
    
    return rates

def jain_fairness(rates):
    """Calculate Jain Fairness Index"""
    if len(rates) == 0 or np.sum(rates) == 0:
        return 0.0
    return (np.sum(rates) ** 2) / (len(rates) * np.sum(rates ** 2))

def objective_function(rates, alpha=0.7):
    """Objective function: alpha*SumRate + (1-alpha)*Jain"""
    sum_rate = np.sum(rates)
    jain = jain_fairness(rates)
    return alpha * sum_rate + (1 - alpha) * jain

def random_assignment(sinr_matrix):
    """Random baseline"""
    num_beams, num_ues = sinr_matrix.shape
    return np.random.randint(0, num_beams, num_ues)

def max_sinr_assignment(sinr_matrix):
    """Max-SINR baseline"""
    return np.argmax(sinr_matrix, axis=0)

def ga_algorithm(sinr_matrix, pop_size=50, generations=100, alpha=0.7):
    """Genetic Algorithm"""
    num_beams, num_ues = sinr_matrix.shape
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    population = [np.random.randint(0, num_beams, num_ues) for _ in range(pop_size)]
    
    for gen in range(generations):
        scores = [evaluate(ind) for ind in population]
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx[:pop_size//2]]
        
        while len(population) < pop_size:
            p1 = population[np.random.randint(0, len(population)//2)]
            p2 = population[np.random.randint(0, len(population)//2)]
            cp = np.random.randint(1, num_ues)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if np.random.random() < 0.1:
                child[np.random.randint(0, num_ues)] = np.random.randint(0, num_beams)
            population.append(child)
    
    return max(population, key=evaluate)

def hga_algorithm(sinr_matrix, pop_size=50, generations=100, alpha=0.7):
    """Hybrid Genetic Algorithm (GA + Local Search)"""
    num_beams, num_ues = sinr_matrix.shape
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    base = np.argmax(sinr_matrix, axis=0)
    population = [np.clip(base + np.random.randint(-1, 2, num_ues), 0, num_beams-1) 
                  for _ in range(pop_size)]
    
    for gen in range(generations):
        scores = [evaluate(ind) for ind in population]
        sorted_idx = np.argsort(scores)[::-1]
        population = [population[i] for i in sorted_idx[:pop_size//2]]
        
        while len(population) < pop_size:
            p1 = population[np.random.randint(0, len(population)//2)]
            p2 = population[np.random.randint(0, len(population)//2)]
            cp = np.random.randint(1, num_ues)
            child = np.concatenate([p1[:cp], p2[cp:]])
            if np.random.random() < 0.1:
                child[np.random.randint(0, num_ues)] = np.random.randint(0, num_beams)
            population.append(child)
        
        # Local search
        for idx in np.random.choice(len(population), max(1, len(population)//3), replace=False):
            ind = population[idx]
            for _ in range(10):
                ue = np.random.randint(num_ues)
                neighbor = ind.copy()
                neighbor[ue] = np.random.randint(0, num_beams)
                if evaluate(neighbor) > evaluate(ind):
                    population[idx] = neighbor
                    ind = neighbor
    
    return max(population, key=evaluate)

def pbig_algorithm(sinr_matrix, max_iter=100, alpha=0.7, d_ratio=0.3):
    """Population-Based Iterated Greedy with Destruction & Reconstruction"""
    num_beams, num_ues = sinr_matrix.shape
    pop_size = 20
    
    def evaluate(ind):
        rates = compute_rates(sinr_matrix, ind)
        return objective_function(rates, alpha)
    
    def greedy_assign(ue_id, current_assignment, destroyed_ues):
        """Greedy reconstruction: assign UE to beam that maximizes objective"""
        best_beam = 0
        best_fitness = -float('inf')
        
        for beam in range(num_beams):
            # Try assigning this UE to this beam
            test_assignment = current_assignment.copy()
            test_assignment[ue_id] = beam
            
            # Calculate fitness if we assign this UE
            fitness = evaluate(test_assignment)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_beam = beam
        
        return best_beam
    
    # Initialize population (random + repaired)
    population = [np.random.randint(0, num_beams, num_ues) for _ in range(pop_size)]
    
    for iteration in range(max_iter):
        # Select a random solution from population
        idx = np.random.randint(0, pop_size)
        S = population[idx].copy()
        
        # Destruction Phase: remove d_ratio of users
        num_destroy = max(1, int(num_ues * d_ratio))
        destroyed_ues = np.random.choice(num_ues, num_destroy, replace=False)
        
        # Mark destroyed UEs as unassigned (-1)
        for ue in destroyed_ues:
            S[ue] = -1
        
        # Reconstruction Phase: greedy reassignment
        for ue in destroyed_ues:
            S[ue] = greedy_assign(ue, S, destroyed_ues)
        
        # Accept if better
        if evaluate(S) > evaluate(population[idx]):
            population[idx] = S
    
    # Return best solution from population
    return max(population, key=evaluate)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.title("⚙️ Simülasyon")

# ========== LENA CONFIG SECTION (Outside Form) ==========
st.sidebar.markdown("**📡 LENA PHY**")

# Temel parametreler (daha kompakt)
lena_col1, lena_col2 = st.sidebar.columns(2)
with lena_col1:
    num_beams = st.selectbox("Beams", options=[4, 8, 16], index=1, key="lena_beams", label_visibility="visible")
with lena_col2:
    num_gnbs = st.selectbox("gNBs", options=[1, 2, 3, 4, 5, 6, 7], index=2, key="lena_gnbs", label_visibility="visible")

# UE sayıları seçimi (daha kısa)
all_ue_options = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 500]
default_ue_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
lena_ue_counts = st.sidebar.multiselect(
    "👥 UE Sayıları",
    options=all_ue_options,
    default=default_ue_counts
)

# Gelişmiş LENA parametreleri (expander içinde - kapalı)
with st.sidebar.expander("🔧 Gelişmiş", expanded=False):
    lena_col3, lena_col4 = st.columns(2)
    with lena_col3:
        lena_tx_power = st.number_input("Tx (dBm)", 30, 50, 46, 2)
        lena_bandwidth = st.selectbox("BW (MHz)", [20, 40, 60, 80, 100], index=4)
    with lena_col4:
        lena_sim_time = st.selectbox("Time (ms)", [200, 300, 400, 500, 1000], index=3)
        lena_numerology_val = st.selectbox("SCS", [(0, "15kHz"), (1, "30kHz"), (2, "60kHz")], index=1, format_func=lambda x: x[1])[0]
    
    lena_frequency_val = st.selectbox(
        "Frekans",
        [(3.5, "3.5 GHz (n78)"), (3.7, "3.7 GHz"), (28.0, "28 GHz (mmW)")],
        index=0,
        format_func=lambda x: x[1]
    )[0]

# LENA konfigürasyonunu session state'e kaydet
st.session_state['lena_config'] = {
    'ue_counts': lena_ue_counts,
    'tx_power': lena_tx_power if 'lena_tx_power' in dir() else 46,
    'bandwidth': lena_bandwidth if 'lena_bandwidth' in dir() else 100,
    'numerology': lena_numerology_val if 'lena_numerology_val' in dir() else 1,
    'sim_time': lena_sim_time if 'lena_sim_time' in dir() else 500,
    'frequency': lena_frequency_val if 'lena_frequency_val' in dir() else 3.5
}

# Re-run LENA button (daha küçük)
rerun_lena = st.sidebar.button("🔄 LENA Sim.", use_container_width=True, type="secondary")

# Mevcut LENA verisi kontrol fonksiyonu
def check_existing_lena_data(beam_count, ue_counts):
    """Hangi UE sayıları için veri mevcut kontrol et"""
    if beam_count == 4:
        data_dir = LENA_OUTPUT_DIR / 'lena_4beam_20251226_024646'
    elif beam_count == 16:
        data_dir = LENA_OUTPUT_DIR / 'lena_16beam_10to100'
    else:
        data_dir = LENA_OUTPUT_DIR / 'lena_8beam_20251226_050004'
    
    existing = []
    missing = []
    
    for ue in ue_counts:
        ue_dir = data_dir / f'ue{ue}'
        if ue_dir.exists() and list(ue_dir.glob('sinr_gnb*.json')):
            existing.append(ue)
        else:
            missing.append(ue)
    
    return existing, missing, data_dir

st.sidebar.markdown("---")

# ========== SIMULATION FORM (Only runs on button click) ==========
with st.sidebar.form(key="simulation_form"):
    st.markdown("**🔬 Algoritmalar**")
    
    # Kullanıcı sayısı artık LENA simülasyonundan otomatik geliyor
    num_ues = 10  # Default değer (harita görüntüleme için)
    
    ric_col1, ric_col2 = st.columns(2)
    with ric_col1:
        max_ues_per_beam = st.checkbox("Kapasite Limit", value=True)  # Default seçili
    with ric_col2:
        if max_ues_per_beam:
            capacity = st.number_input("Max/Beam", 2, 10, 4, label_visibility="visible")
        else:
            capacity = None
    
    # RIC Ayarları (daha kompakt)
    alpha = st.slider("α (TP/Adalet)", 0.0, 1.0, 0.7, 0.1)
    interference_factor = st.slider("İnterferans", 0.0, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    
    # Submit button inside form
    run_simulation = st.form_submit_button("🔬 Algoritmaları Karşılaştır", use_container_width=True, type="primary")

# Fixed algorithms (always run all 4)
selected_algs = ["HGA", "PBIG", "GA", "Max-SINR"]

# Clear button outside form (daha küçük)
clear_results = st.sidebar.button("🔄 Temizle", use_container_width=True)

# ============================================================
# MAIN CONTENT
# ============================================================
st.title("NR Baz İstasyonu ve Hüzme Atama Simülasyonu")
st.markdown("**Huzme-Kullanıcı Atama Optimizasyonunun Gerçek Zamanlı Görselleştirmesi**")

if clear_results:
    st.session_state.clear()
    st.rerun()

# Handle LENA re-run button
if rerun_lena:
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []
    
    # Mevcut simülasyonları kontrol et
    existing_sims = get_existing_lena_simulations(num_beams)
    
    if existing_sims:
        # Mevcut simülasyonlar var - seçim ekranı göster
        st.session_state['show_sim_selection'] = True
        st.session_state['existing_sims'] = existing_sims
    else:
        # Hiç simülasyon yok, yeni oluştur
        folder_name, folder_path = create_timestamped_folder(num_beams)
        st.session_state['current_lena_folder'] = folder_name
        st.session_state['lena_running'] = True
        st.session_state['lena_step'] = 0
        st.session_state['show_sim_selection'] = False
        st.session_state['prev_beams'] = num_beams
        st.session_state['prev_gnbs'] = num_gnbs
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 🔄 Yeni LENA simülasyonu başlatılıyor...")
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📁 Klasör: {folder_name}")
    st.rerun()

# Mevcut simülasyonları seçme ekranı
if st.session_state.get('show_sim_selection', False):
    existing_sims = st.session_state.get('existing_sims', [])
    
    st.info(f"""
    📂 **Mevcut LENA Simülasyonları ({num_beams} Beam)**
    
    Aşağıdaki tarihlerden birini seçebilir, silebilir veya yeni simülasyon koşabilirsiniz:
    """)
    
    # Silinecek klasörü işle
    if st.session_state.get('folder_to_delete', None):
        folder_path = Path(st.session_state['folder_to_delete'])
        folder_name = folder_path.name
        if folder_path.exists():
            try:
                shutil.rmtree(folder_path)
                st.success(f"✅ Klasör silindi: {folder_name}")
            except Exception as e:
                st.error(f"❌ Silme hatası: {e}")
        st.session_state['folder_to_delete'] = None
        # Simülasyon listesini yenile
        st.session_state['existing_sims'] = get_existing_lena_simulations(num_beams)
        existing_sims = st.session_state['existing_sims']
    
    # Mevcut simülasyonları tablo olarak göster
    for idx, sim in enumerate(existing_sims):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            with col1:
                st.markdown(f"**📅 {sim['date']}**")
                st.caption(f"`{sim['name']}`")
            with col2:
                if sim['ue_counts']:
                    st.markdown(f"UE: {min(sim['ue_counts'])}-{max(sim['ue_counts'])} ({len(sim['ue_counts'])} adet)")
                else:
                    st.markdown("UE: Yok")
            with col3:
                if st.button("✅ Kullan", key=f"use_sim_{idx}", use_container_width=True):
                    st.session_state['show_sim_selection'] = False
                    st.session_state['selected_lena_folder'] = sim['name']
                    st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ✅ Seçilen simülasyon: {sim['name']}")
                    st.rerun()
            with col4:
                if st.button("🗑️ Sil", key=f"del_sim_{idx}", use_container_width=True, type="secondary"):
                    st.session_state['folder_to_delete'] = sim['path']
                    st.rerun()
        st.divider()
    
    # Yeni simülasyon butonu
    st.markdown("---")
    if st.button("🆕 Yeni Simülasyon Koş (Tarihli Klasör Oluştur)", use_container_width=True, type="primary"):
        folder_name, folder_path = create_timestamped_folder(num_beams)
        st.session_state['current_lena_folder'] = folder_name
        st.session_state['show_sim_selection'] = False
        st.session_state['lena_running'] = True
        st.session_state['lena_step'] = 0
        st.session_state['prev_beams'] = num_beams
        st.session_state['prev_gnbs'] = num_gnbs
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 🆕 Yeni LENA simülasyonu başlatılıyor...")
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📁 Yeni klasör: {folder_name}")
        st.rerun()
    
    if st.button("❌ İptal", use_container_width=True):
        st.session_state['show_sim_selection'] = False
        st.rerun()

if run_simulation:
    st.session_state['running'] = True
    st.session_state['selected_algs'] = list(selected_algs)
    st.session_state['logs'] = []  # Initialize logs
    
    # Seçili LENA klasörünü kontrol et veya mevcut en yeni simülasyonu bul
    selected_folder = st.session_state.get('selected_lena_folder', None)
    if selected_folder:
        lena_data_dir = selected_folder
    else:
        # Mevcut simülasyonlardan en yeniyi seç
        existing_sims = get_existing_lena_simulations(num_beams)
        if existing_sims:
            lena_data_dir = existing_sims[0]['name']  # En yeni
        else:
            # Hiç simülasyon yok
            lena_data_dir = f'lena_{num_beams}beam_20251226_050004' if num_beams == 8 else f'lena_{num_beams}beam_10to100'
    
    # Add initial log
    st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 🚀 Simülasyon başlatılıyor...")
    st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📊 Parametreler: {num_ues} UE, {num_beams} Beam, {num_gnbs} gNB")
    st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 🧠 Algoritmalar: {', '.join(selected_algs)}")
    
    # SADECE GERÇEK LENA VERİSİ - Sentetik veri yok!
    st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📂 5G-LENA verisi aranıyor: {lena_data_dir}/ue{num_ues}/")
    
    sinr_matrix, actual_ues = load_lena_sinr(lena_data_dir, num_ues)
    
    if sinr_matrix is None:
        # Veri bulunamadı - LENA koşulmalı!
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ❌ HATA: {lena_data_dir}/ue{num_ues} bulunamadı!")
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ⚠️ Lütfen önce 'LENA'yı Tekrar Koştur' butonuna tıklayın.")
        st.session_state['lena_data_missing'] = True
        st.session_state['missing_config'] = {'beams': num_beams, 'ues': num_ues, 'dir': lena_data_dir}
    else:
        # Veri bulundu!
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ✅ SINR matrisi yüklendi [{sinr_matrix.shape[0]}x{sinr_matrix.shape[1]}]")
        st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📝 Gerçek UE sayısı: {actual_ues}")
        st.session_state['sinr_matrix'] = sinr_matrix
        st.session_state['lena_data_missing'] = False
    
    # Store current config
    st.session_state['prev_beams'] = num_beams
    st.session_state['prev_gnbs'] = num_gnbs
    st.session_state['num_ues'] = num_ues
    st.session_state['num_beams'] = num_beams
    st.session_state['lena_data_dir'] = lena_data_dir  # Store for later use
    st.rerun()

# LENA verisi eksik uyarısı
if st.session_state.get('lena_data_missing', False):
    missing = st.session_state.get('missing_config', {})
    st.error(f"""
    ❌ **LENA Verisi Bulunamadı!**
    
    **İstenen konfigürasyon:**
    - Beam: {missing.get('beams', '?')} | UE: {missing.get('ues', '?')}
    - Klasör: `{missing.get('dir', '?')}/ue{missing.get('ues', '?')}/`
    
    **Çözüm:** Sol panelden '**LENA'yı Tekrar Koştur**' butonuna tıklayın.
    """)

# Display LENA running warning if needed
if st.session_state.get('lena_running', False):
    st.markdown("---")
    lena_container = st.container()
    with lena_container:
        lena_col1, lena_col2 = st.columns([5, 1])
        with lena_col1:
            st.warning("""🔄 **5G-LENA PHY Simülasyonu Çalışıyor!**  
            ns-3/5G-LENA ile gerçek SINR matrisleri üretiliyor...""")
        with lena_col2:
            if st.button("⏹️ Durdur", type="secondary", use_container_width=True, key="lena_stop"):
                st.session_state['lena_running'] = False
                st.session_state['lena_stopped'] = True
                st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ⏹️ LENA simülasyonu kullanıcı tarafından durduruldu.")
                st.rerun()
        
        # Kullanıcının seçtiği UE sayılarını al (veya default)
        lena_config = st.session_state.get('lena_config', {})
        ue_targets = sorted(lena_config.get('ue_counts', [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
        
        # LENA parametrelerini al
        lena_tx_power = lena_config.get('tx_power', 46)
        lena_bandwidth = lena_config.get('bandwidth', 100)
        lena_numerology = lena_config.get('numerology', 1)
        lena_sim_time = lena_config.get('sim_time', 500)
        lena_frequency = lena_config.get('frequency', 3.5)
        
        # Yeni tarihli klasörü kullan (session state'den)
        output_subdir = st.session_state.get('current_lena_folder', None)
        if not output_subdir:
            # Fallback: tarihli klasör oluştur
            output_subdir, _ = create_timestamped_folder(num_beams)
            st.session_state['current_lena_folder'] = output_subdir
        
        # Konfigürasyon özetini göster
        st.info(f"""📝 **LENA Konfigürasyonu:**
        - Beam: {num_beams} | gNB: {num_gnbs}
        - Tx Gücü: {lena_tx_power} dBm | Bant: {lena_bandwidth} MHz
        - Numerology: {lena_numerology} | Süre: {lena_sim_time}ms
        - UE Sayıları: {', '.join(map(str, ue_targets))}""")
        
        # Get current step from session state
        lena_step = st.session_state.get('lena_step', 0)
        
        # TÜM SİMÜLASYONLARI TEK SEFERDE KOŞ (st.rerun() olmadan)
        if lena_step < len(ue_targets):
            # Progress bar placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            completed_placeholder = st.empty()
            
            # Tüm UE'ler için simülasyon koş (blocking loop)
            for step_idx in range(lena_step, len(ue_targets)):
                current_ue = ue_targets[step_idx]
                
                # Progress bar güncelle
                progress_val = step_idx / len(ue_targets)
                progress_placeholder.progress(progress_val, text=f"İlerleme: {int(progress_val * 100)}% ({step_idx}/{len(ue_targets)})")
                
                # Status güncelle
                status_placeholder.warning(f"**🔄 {current_ue} UE için ns-3/5G-LENA simülasyonu çalışıyor... (Bu 20-60 saniye sürebilir)**")
                
                # Tamamlananları göster
                completed_ues = ue_targets[:step_idx]
                if completed_ues:
                    completed_placeholder.success(f"✅ Tamamlanan: {', '.join(map(str, completed_ues))} UE")
                
                # GERÇEK LENA SİMÜLASYONU ÇALIŞTIR
                success, actual_ues, message = run_lena_simulation(
                    num_ues=current_ue,
                    num_beams=num_beams,
                    num_gnbs=num_gnbs,
                    output_subdir=output_subdir,
                    tx_power=lena_tx_power,
                    bandwidth=lena_bandwidth,
                    numerology=lena_numerology,
                    sim_time=lena_sim_time,
                    frequency=lena_frequency
                )
                
                # Log ekle
                st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] {message}")
                
                # Session state'i güncelle (her adımda)
                st.session_state['lena_step'] = step_idx + 1
            
            # TÜM sİMÜLASYONLAR TAMAMLANDI
            progress_placeholder.progress(1.0, text=f"İlerleme: 100% ({len(ue_targets)}/{len(ue_targets)})")
            status_placeholder.empty()
            completed_placeholder.success(f"✅ Tamamlanan: {', '.join(map(str, ue_targets))} UE")
            
            st.session_state['lena_running'] = False
            st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ✅ Tüm LENA simülasyonları tamamlandı!")
            st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📁 SINR verileri: {output_subdir}/")
            
            # Benchmark CSV'yi yeniden oluştur
            st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] 📊 Benchmark CSV oluşturuluyor...")
            csv_success, csv_file = regenerate_benchmark_csv(output_subdir, num_beams, ue_targets)
            if csv_success:
                st.session_state['logs'].append(f"[{time.strftime('%H:%M:%S')}] ✅ {csv_file} oluşturuldu!")
            
            time.sleep(1)
            st.rerun()
        else:
            # Tüm UE'ler tamamlandı
            st.success(f"✅ **Tüm LENA simülasyonları tamamlandı!**\n\n"
                      f"📁 SINR verileri: `{output_subdir}/`\n\n"
                      f"📊 Toplam: {len(ue_targets)} farklı UE sayısı")
            st.session_state['lena_running'] = False
            time.sleep(1)
            st.rerun()
    
    st.markdown("---")

# Display results if available
if 'sinr_matrix' in st.session_state:
    sinr_matrix = st.session_state['sinr_matrix']
    num_ues = st.session_state['num_ues']
    num_beams = st.session_state['num_beams']
    
    # Create layout: Controls on left, Map on right
    results_col, map_col = st.columns([1, 1])
    
    # Generate gNB and UE positions (Bornova, İzmir koordinatları)
    np.random.seed(42)
    
    # Bornova merkez koordinatları
    center_lat = 38.4628
    center_lon = 27.2156
    
    # gNB pozisyonları (Bornova bölgesinde üçgen yerleşim)
    gnb_positions = [
        (38.4670, 27.2100, 'gNB-0', '#3b82f6'),   # Kuzey-Batı (Mavi)
        (38.4670, 27.2250, 'gNB-1', '#10b981'),   # Kuzey-Doğu (Yeşil)
        (38.4570, 27.2175, 'gNB-2', '#8b5cf6'),   # Güney (Mor)
    ]
    
    # UE pozisyonları (rastgele dağılım)
    # Get available UE counts for map visualization
    available_ue_counts = lena_ue_counts if lena_ue_counts else [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Determine display UE count (use selected if available, otherwise default)
    if 'selected_map_ue' in st.session_state:
        display_num_ues = st.session_state['selected_map_ue']
    else:
        display_num_ues = num_ues
    
    ue_positions = []
    # Adjust UE distribution area based on UE count for better visibility
    if display_num_ues <= 30:
        lat_range, lon_range = 0.008, 0.012  # Small area for few UEs
    elif display_num_ues <= 60:
        lat_range, lon_range = 0.012, 0.018  # Medium area
    else:
        lat_range, lon_range = 0.018, 0.025  # Large area for 100 UEs
    
    for i in range(display_num_ues):
        lat = center_lat + np.random.uniform(-lat_range, lat_range)
        lon = center_lon + np.random.uniform(-lon_range, lon_range)
        ue_positions.append((lat, lon))
    
    # ========== RIGHT: MAP VISUALIZATION ==========
    with map_col:
        st.subheader("🗺️ Baz İstasyonu ve Kullanıcı Lokasyonları")
        
        # Create folium map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,  # More zoomed out for better overview
            tiles='OpenStreetMap'
        )
        
        # Beam renkleri (8 farklı renk)
        beam_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                       '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
                       '#BB8FCE', '#85C1E9', '#F8B500', '#00CED1',
                       '#FF69B4', '#32CD32', '#FFD700', '#8A2BE2']
        
        # Fonksiyon: Sektör (wedge) koordinatlarını oluştur
        def create_sector_coords(center_lat, center_lon, radius_km, start_angle, end_angle, num_points=20):
            """Create polygon coordinates for a sector/wedge shape"""
            coords = [(center_lat, center_lon)]  # Center point
            
            # Convert radius from km to degrees (approximate)
            lat_per_km = 1 / 111.0  # ~111 km per degree latitude
            lon_per_km = 1 / (111.0 * np.cos(np.radians(center_lat)))  # Adjust for latitude
            
            for angle in np.linspace(start_angle, end_angle, num_points):
                rad = np.radians(angle)
                dlat = radius_km * lat_per_km * np.cos(rad)
                dlon = radius_km * lon_per_km * np.sin(rad)
                coords.append((center_lat + dlat, center_lon + dlon))
            
            coords.append((center_lat, center_lon))  # Close the polygon
            return coords
        
        # Add gNBs with beam sectors
        for gnb_idx, (lat, lon, name, gnb_color) in enumerate(gnb_positions):
            sector_angle = 360 / num_beams  # Her beam için açı
            
            # Draw beam sectors
            for beam_id in range(num_beams):
                start_angle = beam_id * sector_angle - 90  # Start from north
                end_angle = start_angle + sector_angle
                
                # Sector coordinates
                sector_coords = create_sector_coords(
                    lat, lon, 
                    radius_km=0.8,  # 800m radius
                    start_angle=start_angle, 
                    end_angle=end_angle
                )
                
                beam_color = beam_colors[beam_id % len(beam_colors)]
                
                # Draw sector polygon
                folium.Polygon(
                    locations=sector_coords,
                    color=beam_color,
                    weight=2,
                    fill=True,
                    fill_color=beam_color,
                    fill_opacity=0.3,
                    popup=f'{name} - Beam {beam_id}',
                    tooltip=f'{name} Beam-{beam_id}'
                ).add_to(m)
            
            # gNB center marker (tower icon)
            folium.Marker(
                location=[lat, lon],
                popup=f'<b>{name}</b><br>Beams: {num_beams}<br>Power: 46 dBm',
                icon=folium.Icon(color='red', icon='tower-broadcast', prefix='fa'),
                tooltip=name
            ).add_to(m)
        
        # Add UEs
        if 'assignment' in st.session_state and len(st.session_state['assignment']) == display_num_ues:
            assignment = st.session_state['assignment']
            for ue_id, (lat, lon) in enumerate(ue_positions):
                beam_id = assignment[ue_id]
                color = beam_colors[beam_id % len(beam_colors)]
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=f'<b>UE-{ue_id}</b><br>Beam: {beam_id}',
                    tooltip=f'UE-{ue_id} → Beam-{beam_id}'
                ).add_to(m)
        else:
            for ue_id, (lat, lon) in enumerate(ue_positions):
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=5,
                    color='blue',
                    fill=True,
                    fill_color='blue',
                    fill_opacity=0.6,
                    popup=f'UE-{ue_id}',
                    tooltip=f'UE-{ue_id}'
                ).add_to(m)
        
        # Display map (returned_objects=[] prevents map interactions from triggering reruns)
        st_folium(m, width=None, height=580, use_container_width=True, returned_objects=[])
    
    # ========== LEFT: RESULTS & METRICS ==========
    with results_col:
        st.subheader("Algoritma Karşılaştırma Sonuçları")
        
        if 'sinr_matrix' in st.session_state and 'selected_algs' in st.session_state:
            # Get stored algorithms
            algs_to_run = st.session_state['selected_algs']
            
            if len(algs_to_run) == 0:
                st.warning("⚠️ Lütfen en az bir algoritma seçin!")
            else:
                # Status container
                status_container = st.container()
                with status_container:
                    st.markdown("##### ⏳ Simülasyon Durumu")
                    progress_bar = st.progress(0, text="Başlatılıyor...")
                    status_text = st.empty()
                
                # Run algorithms
                results_data = []
                logs = st.session_state.get('logs', [])
                
                best_score_alg = None
                best_score = -float('inf')
                best_fair_alg = None
                best_fair = -float('inf')
                
                for alg_idx, alg_name in enumerate(algs_to_run):
                    progress = (alg_idx + 1) / len(algs_to_run)
                    progress_bar.progress(progress, text=f"🔄 {alg_name} çalıştırılıyor...")
                    status_text.info(f"🧠 Algoritma: **{alg_name}** | İlerleme: {int(progress*100)}%")
                    
                    # Add to logs
                    logs.append(f"[{time.strftime('%H:%M:%S')}] 🔄 {alg_name} başlatılıyor...")
                    
                    start_time = time.time()
                    
                    if alg_name == 'GA':
                        assignment = ga_algorithm(sinr_matrix, alpha=alpha)
                    elif alg_name == 'HGA':
                        assignment = hga_algorithm(sinr_matrix, alpha=alpha)
                    elif alg_name == 'PBIG':
                        assignment = pbig_algorithm(sinr_matrix, alpha=alpha)
                    else:  # Max-SINR
                        assignment = max_sinr_assignment(sinr_matrix)
                    
                    runtime = time.time() - start_time
                    rates = compute_rates(sinr_matrix, assignment, interference_factor)
                    
                    sum_rate = np.sum(rates)
                    jain = jain_fairness(rates)
                    fitness = objective_function(rates, alpha)
                    throughput_mbps = sum_rate * 100 * 0.8
                    
                    # Track best
                    if fitness > best_score:
                        best_score = fitness
                        best_score_alg = alg_name
                    if jain > best_fair:
                        best_fair = jain
                        best_fair_alg = alg_name
                    
                    results_data.append({
                        'alg': alg_name,
                        'sum_rate': sum_rate,
                        'throughput': throughput_mbps,
                        'jain': jain,
                        'fitness': fitness,
                        'runtime': runtime * 1000
                    })
                    
                    # Log completion
                    logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ {alg_name} tamamlandı ({runtime*1000:.1f}ms)")
                    
                    # Store first assignment for visualization
                    if alg_idx == 0:
                        st.session_state['assignment'] = assignment
                
                # Update status on completion
                progress_bar.progress(1.0, text="✅ Tüm algoritmalar tamamlandı!")
                status_text.success(f"🏆 En iyi: **{best_score_alg}** | Skor: {best_score:.2f}")
                
                # Store logs
                st.session_state['logs'] = logs
            
                # UE selection for table update
                available_ue_counts = lena_ue_counts if lena_ue_counts else [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                
                # UE selector above table
                ue_select_col1, ue_select_col2 = st.columns([2, 1])
                with ue_select_col1:
                    selected_table_ue = st.selectbox(
                        "📈 Tablo için UE sayısını seçin:",
                        options=available_ue_counts,
                        index=0 if num_ues not in available_ue_counts else available_ue_counts.index(num_ues),
                        key="results_table_ue_selector"
                    )
                with ue_select_col2:
                    st.metric("Seçili UE", selected_table_ue)
                
                # Update session state for map synchronization
                st.session_state['selected_map_ue'] = selected_table_ue
                
                # Check if we need to recalculate for different UE count
                if selected_table_ue != num_ues:
                    # Load from CSV instead of recalculating
                    benchmark_file = f"results_{num_beams}beam_10to100.csv"
                    
                    if os.path.exists(benchmark_file):

                        df_bench = pd.read_csv(benchmark_file)
                        df_selected = df_bench[df_bench['num_ues'] == selected_table_ue]
                        
                        if len(df_selected) == 0:
                            st.error(f"❌ {selected_table_ue} UE için CSV'de veri bulunamadı!")
                            results_data = []
                        else:
                            # Build results from CSV with fixed order
                            algorithm_order = ['HGA', 'PBIG', 'GA', 'Max-SINR']
                            results_data = []
                            best_score = -float('inf')
                            best_score_alg = None
                            best_fair = -float('inf')
                            best_fair_alg = None
                            
                            # Process in fixed order
                            for alg_name in algorithm_order:
                                row_data = df_selected[df_selected['algorithm'] == alg_name]
                                if len(row_data) > 0:
                                    row = row_data.iloc[0]
                                    sum_rate = row['sum_rate']
                                    throughput = row['throughput']
                                    jain = row['fairness']
                                    
                                    # Calculate fitness (same formula as algorithms)
                                    fitness = alpha * sum_rate + (1 - alpha) * jain
                                    
                                    if fitness > best_score:
                                        best_score = fitness
                                        best_score_alg = alg_name
                                    if jain > best_fair:
                                        best_fair = jain
                                        best_fair_alg = alg_name
                                    
                                    results_data.append({
                                        'alg': alg_name,
                                        'sum_rate': sum_rate,
                                        'throughput': throughput,
                                        'jain': jain,
                                        'fitness': fitness,
                                        'runtime': 0  # CSV doesn't have runtime
                                    })
                    else:
                        st.error(f"❌ Benchmark dosyası bulunamadı: {benchmark_file}")
                        results_data = []
                
                # Update num_ues for table display
                display_num_ues = selected_table_ue
                
                # Build custom HTML table matching the design
                best_score_display = f"{best_score:.2f}" if best_score < 1000000 else f"{best_score:.0f}"
                best_fair_display = f"{best_fair:.2f}"
                
                table_html = f'''
                <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #0d1a2d; padding: 20px; border-radius: 8px;">
                    <!-- Summary Labels -->
                    <div style="margin-bottom: 15px;">
                        <span style="background-color: #1a3a5c; color: #4fc3f7; padding: 8px 16px; border-radius: 4px; font-size: 13px; margin-right: 10px;">
                            En iyi skor: {best_score_alg} ({best_score_display})
                        </span>
                        <span style="background-color: #1a3a5c; color: #81c784; padding: 8px 16px; border-radius: 4px; font-size: 13px;">
                            En adil: {best_fair_alg} (Jain={best_fair_display})
                        </span>
                    </div>
                    
                    <!-- Table -->
                    <table style="width: 100%; border-collapse: collapse; background-color: #0d1a2d;">
                        <thead>
                            <tr style="border-bottom: 1px solid #2a4a6c;">
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">YÖNTEM</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">DURUM</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">TOPLAM MBPS</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">JAIN</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">SKOR</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">USERS</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">BEAMS</th>
                                <th style="padding: 12px 10px; text-align: left; color: #7a8a9a; font-size: 11px; font-weight: 600; text-transform: uppercase;">GNBS</th>
                            </tr>
                        </thead>
                        <tbody>
                '''
                
                for r in results_data:
                    is_best = r['alg'] == best_score_alg
                    if is_best:
                        badge = '<span style="background-color: #fbbf24; color: #000; padding: 3px 10px; border-radius: 3px; font-size: 11px; font-weight: 600;">BEST</span>'
                    else:
                        badge = '<span style="background-color: #1e5631; color: #4ade80; padding: 3px 10px; border-radius: 3px; font-size: 11px; font-weight: 500;">Feasible</span>'
                    
                    skor_display = f"{r['fitness']:.2f}" if r['fitness'] < 1000000 else f"{r['fitness']:.0f}"
                    
                    table_html += f'''
                            <tr style="border-bottom: 1px solid #1a3050;">
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">{r['alg']}</td>
                                <td style="padding: 12px 10px;">{badge}</td>
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">{r['throughput']:.2f}</td>
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">{r['jain']:.2f}</td>
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">{skor_display}</td>
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">{display_num_ues}</td>
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">{num_beams}</td>
                                <td style="padding: 12px 10px; color: #e0e0e0; font-size: 13px;">3</td>
                            </tr>
                    '''
                
                table_html += '''
                        </tbody>
                    </table>
                </div>
                '''
                
                # Render with components.html for proper styling
                # Height: header(60) + summary(50) + rows(45 each) + padding(40)
                table_height = 150 + len(results_data) * 50
                components.html(table_html, height=table_height, scrolling=False)
                
                # Runtime info
                total_runtime = sum([r['runtime'] for r in results_data])
                st.caption(f"⏱️ Toplam Çalışma Süresi: {total_runtime:.1f} ms")
    
    # ========== UE SWEEP COMPARISON CHARTS (10-100 UE) ==========
    st.divider()
    
    # Capacity selection dropdown
    capacity_scenario = st.selectbox(
        "📊 Kapasite Senaryosu",
        options=["Sınırsız", "Max 4 UE/Beam"],
        index=0,
        key="capacity_scenario_selector"
    )
    
    # Determine which benchmark file to use based on beam count AND capacity
    if num_beams == 4:
        if capacity_scenario == "Max 4 UE/Beam":
            benchmark_file = "results_4beam_10to100_cap4.csv"
        else:
            benchmark_file = "results_4beam_10to100.csv"
        chart_beam_label = "4 Hüzme"
    elif num_beams == 16:
        if capacity_scenario == "Max 4 UE/Beam":
            benchmark_file = "results_16beam_10to100_cap4.csv"
        else:
            benchmark_file = "results_16beam_10to100.csv"
        chart_beam_label = "16 Hüzme"
    else:
        if capacity_scenario == "Max 4 UE/Beam":
            benchmark_file = "results_8beam_10to100_cap4.csv"
        else:
            benchmark_file = "results_8beam_10to100.csv"
        chart_beam_label = "8 Hüzme"
    
    # Update title based on capacity
    if capacity_scenario == "Max 4 UE/Beam":
        st.subheader(f"📈 Algoritma Performans Karşılaştırması ({chart_beam_label}, 10-100 UE, Max 4 UE/Beam)")
    else:
        st.subheader(f"📈 Algoritma Performans Karşılaştırması ({chart_beam_label}, 10-100 UE)")
    
    # Load benchmark results
    chart_beam_display = num_beams  # Will be updated if fallback
    try:
        df_bench = pd.read_csv(benchmark_file)
        chart_beam_display = num_beams  # File found, use selected beam count
        
        # Prepare data for plotting
        algorithms = df_bench['algorithm'].unique()
        ue_counts = sorted(df_bench['num_ues'].unique())
        
        # Color palette for algorithms (matching reference image)
        algo_colors = {
            'Max-SINR': '#8B008B',  # Purple (dashed)
            'HGA': '#DC143C',       # Red
            'GA': '#FFA500',        # Orange
            'PBIG': '#0000CD'       # Blue
        }
        algo_markers = {
            'Max-SINR': 'D',
            'HGA': 's',
            'GA': 'o',
            'PBIG': '^'
        }
        
        # Create 3 charts side by side
        chart1, chart2, chart3 = st.columns(3)
        
        # High quality matplotlib charts
        plt.rcParams['figure.dpi'] = 200
        plt.rcParams['savefig.dpi'] = 200
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['grid.color'] = '#cccccc'
        
        fig_width, fig_height = 10, 6
        tech_subtitle = "5G-LENA PHY, n78 3.5GHz, 100 MHz"
        
        # Color palette for algorithms
        algo_colors = {
            'Max-SINR': '#8B008B',
            'HGA': '#DC143C',
            'GA': '#FFA500',
            'PBIG': '#0000CD'
        }
        algo_markers = {
            'Max-SINR': 'D',
            'HGA': 's',
            'GA': 'o',
            'PBIG': '^'
        }
        
        # Chart 1: Jain Index
        with chart1:
            fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
            for alg in algorithms:
                alg_data = df_bench[df_bench['algorithm'] == alg].sort_values('num_ues')
                color = algo_colors.get(alg, '#888888')
                marker = algo_markers.get(alg, 'o')
                linestyle = '--' if alg == 'Max-SINR' else '-'
                label = 'Max-SINR' if alg == 'Max-SINR' else alg
                ax1.plot(alg_data['num_ues'], alg_data['fairness'], 
                        marker=marker, label=label, color=color, linewidth=2.5, markersize=8, linestyle=linestyle)
            ax1.set_xlabel('Kullanıcı Sayısı', fontweight='bold')
            ax1.set_ylabel('Jain İndeksi', fontweight='bold')
            ax1.set_title(f'Jain Adalet İndeksi ({chart_beam_display} Beam)\n{tech_subtitle}', fontweight='bold')
            ax1.legend(loc='upper right', framealpha=0.9)
            ax1.grid(True, alpha=0.4, linestyle='-')
            # Autoscale y-axis for better visibility
            ax1.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax1.set_xlim([5, 105])
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close(fig1)
        
        # Chart 2: Total Throughput
        with chart2:
            fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
            for alg in algorithms:
                alg_data = df_bench[df_bench['algorithm'] == alg].sort_values('num_ues')
                color = algo_colors.get(alg, '#888888')
                marker = algo_markers.get(alg, 'o')
                linestyle = '--' if alg == 'Max-SINR' else '-'
                label = 'Max-SINR' if alg == 'Max-SINR' else alg
                ax2.plot(alg_data['num_ues'], alg_data['throughput'], 
                        marker=marker, label=label, color=color, linewidth=2.5, markersize=8, linestyle=linestyle)
            ax2.set_xlabel('Kullanıcı Sayısı', fontweight='bold')
            ax2.set_ylabel('Sistem Toplam Veri Hızı (Mbps)', fontweight='bold')
            ax2.set_title(f'Sistem Toplam Veri Hızı ({chart_beam_display} Beam)\n{tech_subtitle}', fontweight='bold')
            ax2.legend(loc='upper left', framealpha=0.9)
            ax2.grid(True, alpha=0.4, linestyle='-')
            ax2.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax2.set_xlim([5, 105])
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        
        # Chart 3: Objective Function
        with chart3:
            fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
            alpha_val = 0.7
            for alg in algorithms:
                alg_data = df_bench[df_bench['algorithm'] == alg].sort_values('num_ues')
                color = algo_colors.get(alg, '#888888')
                marker = algo_markers.get(alg, 'o')
                linestyle = '--' if alg == 'Max-SINR' else '-'
                max_tp = df_bench['throughput'].max()
                norm_tp = alg_data['throughput'] / max_tp
                fitness = alpha_val * norm_tp + (1 - alpha_val) * alg_data['fairness']
                label = 'Max-SINR' if alg == 'Max-SINR' else alg
                ax3.plot(alg_data['num_ues'], fitness, 
                        marker=marker, label=label, color=color, linewidth=2.5, markersize=8, linestyle=linestyle)
            ax3.set_xlabel('Kullanıcı Sayısı', fontweight='bold')
            ax3.set_ylabel('Amaç Fonksiyonu Skoru', fontweight='bold')
            ax3.set_title(f'Amaç Fonksiyonu ({chart_beam_display} Beam)\n{tech_subtitle}', fontweight='bold')
            ax3.legend(loc='upper right', framealpha=0.9)
            ax3.grid(True, alpha=0.4, linestyle='-')
            ax3.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            ax3.set_xlim([5, 105])
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close(fig3)
            
    except FileNotFoundError:
        st.warning(f"⚠️ {chart_beam_label} için benchmark verisi bulunamadı ({benchmark_file}). 8 hüzme verisi gösteriliyor.")
        chart_beam_display = 8
        try:
            df_bench = pd.read_csv("results_8beam_10to100.csv")
            algorithms = df_bench['algorithm'].unique()
            
            algo_colors = {'Max-SINR': '#8B008B', 'HGA': '#DC143C', 'GA': '#FFA500', 'PBIG': '#0000CD'}
            algo_markers = {'Max-SINR': 'D', 'HGA': 's', 'GA': 'o', 'PBIG': '^'}
            
            chart1, chart2, chart3 = st.columns(3)
            plt.rcParams['figure.dpi'] = 200
            fig_width, fig_height = 10, 6
            tech_subtitle = "5G-LENA PHY, n78 3.5GHz, 100 MHz"
            
            with chart1:
                fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))
                for alg in algorithms:
                    alg_data = df_bench[df_bench['algorithm'] == alg].sort_values('num_ues')
                    color = algo_colors.get(alg, '#888888')
                    marker = algo_markers.get(alg, 'o')
                    linestyle = '--' if alg == 'Max-SINR' else '-'
                    label = 'Max-SINR' if alg == 'Max-SINR' else alg
                    ax1.plot(alg_data['num_ues'], alg_data['fairness'], 
                            marker=marker, label=label, color=color, linewidth=2.5, markersize=8, linestyle=linestyle)
                ax1.set_xlabel('Kullanıcı Sayısı', fontweight='bold')
                ax1.set_ylabel('Jain İndeksi', fontweight='bold')
                ax1.set_title(f'Jain Adalet İndeksi (8 Beam)\n{tech_subtitle}', fontweight='bold')
                ax1.legend(loc='upper right', framealpha=0.9)
                ax1.grid(True, alpha=0.4)
                # Autoscale y-axis for better visibility
                ax1.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                ax1.set_xlim([5, 105])
                plt.tight_layout()
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)
            
            with chart2:
                fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
                for alg in algorithms:
                    alg_data = df_bench[df_bench['algorithm'] == alg].sort_values('num_ues')
                    color = algo_colors.get(alg, '#888888')
                    marker = algo_markers.get(alg, 'o')
                    linestyle = '--' if alg == 'Max-SINR' else '-'
                    label = 'Max-SINR' if alg == 'Max-SINR' else alg
                    ax2.plot(alg_data['num_ues'], alg_data['throughput'], 
                            marker=marker, label=label, color=color, linewidth=2.5, markersize=8, linestyle=linestyle)
                ax2.set_xlabel('Kullanıcı Sayısı', fontweight='bold')
                ax2.set_ylabel('Sistem Toplam Veri Hızı (Mbps)', fontweight='bold')
                ax2.set_title(f'Sistem Toplam Veri Hızı (8 Beam)\n{tech_subtitle}', fontweight='bold')
                ax2.legend(loc='upper left', framealpha=0.9)
                ax2.grid(True, alpha=0.4)
                ax2.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                ax2.set_xlim([5, 105])
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
            
            with chart3:
                fig3, ax3 = plt.subplots(figsize=(fig_width, fig_height))
                alpha_val = 0.7
                for alg in algorithms:
                    alg_data = df_bench[df_bench['algorithm'] == alg].sort_values('num_ues')
                    color = algo_colors.get(alg, '#888888')
                    marker = algo_markers.get(alg, 'o')
                    linestyle = '--' if alg == 'Max-SINR' else '-'
                    max_tp = df_bench['throughput'].max()
                    norm_tp = alg_data['throughput'] / max_tp
                    fitness = alpha_val * norm_tp + (1 - alpha_val) * alg_data['fairness']
                    label = 'Max-SINR' if alg == 'Max-SINR' else alg
                    ax3.plot(alg_data['num_ues'], fitness, 
                            marker=marker, label=label, color=color, linewidth=2.5, markersize=8, linestyle=linestyle)
                ax3.set_xlabel('Kullanıcı Sayısı', fontweight='bold')
                ax3.set_ylabel('Amaç Fonksiyonu Skoru', fontweight='bold')
                ax3.set_title(f'Amaç Fonksiyonu (8 Beam)\n{tech_subtitle}', fontweight='bold')
                ax3.legend(loc='upper right', framealpha=0.9)
                ax3.grid(True, alpha=0.4)
                ax3.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
                ax3.set_xlim([5, 105])
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)
        except:
            st.error("❌ Hiçbir benchmark verisi bulunamadı.")
    
    # ========== DETAILED UE INFORMATION TABLE (EN ALTTA) ==========
    st.divider()
    st.subheader("📊 Detaylı UE Bilgileri")
    
    # Get the selected UE count from session state (from dropdown)
    selected_ue_for_table = st.session_state.get('selected_map_ue', num_ues)
    
    # Check if we have data for the selected UE count
    if 'assignment' in st.session_state and 'sinr_matrix' in st.session_state:
        # If selected UE count differs from original, load appropriate data
        if selected_ue_for_table != num_ues:
            # Get LENA data directory from session state
            lena_data_dir = st.session_state.get('lena_data_dir', f'lena_{num_beams}beam_10to100')
            
            # Load SINR matrix for selected UE count
            table_sinr_matrix, table_actual_ues = load_lena_sinr(lena_data_dir, selected_ue_for_table)
            
            if table_sinr_matrix is None:
                st.warning(f"⚠️ {selected_ue_for_table} UE için LENA verisi bulunamadı. Orijinal simülasyon verisi gösteriliyor.")
                table_sinr_matrix = st.session_state['sinr_matrix']
                table_assignment = st.session_state['assignment']
            else:
                # Run best algorithm (HGA) for selected UE count to get assignment
                table_assignment = hga_algorithm(table_sinr_matrix, alpha=alpha)
        else:
            # Use original simulation data
            table_assignment = st.session_state['assignment']
            table_sinr_matrix = st.session_state['sinr_matrix']
        
        rates = compute_rates(table_sinr_matrix, table_assignment, interference_factor)
        
        # Build UE details data
        ue_details = []
        for ue_id in range(len(table_assignment)):
            beam_id = table_assignment[ue_id]
            
            # Calculate gNB ID from beam ID
            # LENA combines 3 gNBs: gnb0 (beam 0-7), gnb1 (beam 8-15), gnb2 (beam 16-23)
            gnb_id = beam_id // num_beams  # num_beams = beams per gNB (8)
            local_beam_id = beam_id % num_beams  # Local beam within gNB (0-7)
            
            sinr_db = table_sinr_matrix[beam_id, ue_id]
            rate = rates[ue_id]
            throughput_mbps = rate * 100 * 0.8  # Convert to Mbps
            
            # Count co-beam users for interference
            co_beam_users = sum(1 for b in table_assignment if b == beam_id)
            interference_penalty = interference_factor * (co_beam_users - 1)
            effective_sinr = sinr_db - interference_penalty
            
            ue_details.append({
                'UE ID': ue_id,
                'gNB ID': gnb_id,
                'Beam ID': local_beam_id,
                'SINR (dB)': f"{sinr_db:.2f}",
                'Efektif SINR (dB)': f"{effective_sinr:.2f}",
                'Data Rate': f"{rate:.3f}",
                'Throughput (Mbps)': f"{throughput_mbps:.2f}",
                'Co-Beam UEs': co_beam_users,
                'İnterferans (dB)': f"{interference_penalty:.2f}"
            })
        
        # Create DataFrame
        df_ue = pd.DataFrame(ue_details)
        
        # Show which UE count is being displayed
        st.info(f"📊 {len(df_ue)} UE için detaylı bilgiler gösteriliyor (Seçili: {selected_ue_for_table} UE)")
        
        # Display as interactive table with pagination
        st.dataframe(
            df_ue,
            use_container_width=True,
            height=min(600, 35 + len(df_ue) * 35),  # Auto-height with max 600px
            hide_index=True  # Remove the leftmost index column
        )
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Toplam UE", len(df_ue))
        with col2:
            avg_sinr = df_ue['SINR (dB)'].apply(lambda x: float(x)).mean()
            st.metric("📶 Ortalama SINR", f"{avg_sinr:.2f} dB")
        with col3:
            total_throughput = df_ue['Throughput (Mbps)'].apply(lambda x: float(x)).sum()
            st.metric("🚀 Toplam Throughput", f"{total_throughput:.2f} Mbps")
        with col4:
            jain_idx = jain_fairness(rates)
            st.metric("⚖️ Jain Index", f"{jain_idx:.3f}")
        
        # Download button for CSV export
        csv = df_ue.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 UE Verilerini İndir (CSV)",
            data=csv,
            file_name=f'ue_details_{num_ues}ue_{num_beams}beam.csv',
            mime='text/csv',
        )
    else:
        st.info("⚠️ Simülasyon henüz çalıştırılmadı. UE detaylarını görmek için simülasyonu başlatın.")
    
    # ========== TERMINAL-LIKE LOG OUTPUT (EN ALTTA) ==========
    st.divider()
    st.subheader("📟 Simülasyon Log")
    
    logs = st.session_state.get('logs', [])
    
    if logs:
        log_text = "\n".join(logs[-15:])
        st.code(log_text, language="bash")
    else:
        st.code("Henüz log yok. Simülasyonu başlatın...", language="bash")

else:
    st.info("👈 Parametreleri ayarlayın ve **Simülasyonu Başlat** butonuna tıklayın")
