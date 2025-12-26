#!/usr/bin/env python3
"""
Analyze and visualize RIC benchmark results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


def load_results(filenames) -> pd.DataFrame:
    """Load benchmark results from one or more CSV files"""
    if isinstance(filenames, str):
        filenames = [filenames]
    dfs = []
    for filename in filenames:
        try:
            df = pd.read_csv(filename)
            print(f"✓ Loaded {len(df)} results from {filename}")
            dfs.append(df)
        except FileNotFoundError:
            print(f"ERROR: File not found: {filename}")
            print("Run benchmark first: python3 benchmark_scalability.py")
            sys.exit(1)
    if not dfs:
        print("ERROR: No input files loaded")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


def plot_scalability(df: pd.DataFrame, output_dir: str = "."):
    """Create scalability plots"""
    
    # Get unique configurations
    algorithms = df['algorithm'].unique()
    beam_configs = sorted(df['num_beams'].unique())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RIC Algoritmalarının Ölçeklenebilirlik Analizi', fontsize=16, fontweight='bold')
    
    for num_beams in beam_configs:
        df_beams = df[df['num_beams'] == num_beams]
        
        # Plot 1: Objective Value vs UEs
        ax = axes[0, 0]
        for algo in algorithms:
            df_algo = df_beams[df_beams['algorithm'] == algo]
            ax.plot(df_algo['num_ues'], df_algo['avg_objective'], 
                   marker='o', label=f"{algo} ({num_beams}B)", linewidth=2)
            ax.fill_between(df_algo['num_ues'], 
                          df_algo['avg_objective'] - df_algo['std_objective'],
                          df_algo['avg_objective'] + df_algo['std_objective'],
                          alpha=0.2)
        ax.set_xlabel('Kullanıcı Sayısı', fontsize=11)
        ax.set_ylabel('Amaç Fonksiyonu (Toplam Veri Hızı)', fontsize=11)
        ax.set_title('Performans vs Ölçek', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Runtime vs UEs (log scale)
        ax = axes[0, 1]
        for algo in algorithms:
            df_algo = df_beams[df_beams['algorithm'] == algo]
            ax.plot(df_algo['num_ues'], df_algo['avg_runtime_ms'], 
                   marker='s', label=f"{algo} ({num_beams}B)", linewidth=2)
        ax.set_xlabel('Kullanıcı Sayısı', fontsize=11)
        ax.set_ylabel('Çalışma Süresi (ms)', fontsize=11)
        ax.set_title('Hesaplama Karmaşıklığı', fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Average SINR vs UEs
        ax = axes[1, 0]
        for algo in algorithms:
            df_algo = df_beams[df_beams['algorithm'] == algo]
            ax.plot(df_algo['num_ues'], df_algo['avg_sinr_dB'], 
                   marker='^', label=f"{algo} ({num_beams}B)", linewidth=2)
        ax.set_xlabel('Kullanıcı Sayısı', fontsize=11)
        ax.set_ylabel('Ortalama SINR (dB)', fontsize=11)
        ax.set_title('Ortalama UE SINR', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Min SINR vs UEs (Cell-edge performance)
        ax = axes[1, 1]
        for algo in algorithms:
            df_algo = df_beams[df_beams['algorithm'] == algo]
            ax.plot(df_algo['num_ues'], df_algo['min_sinr_dB'], 
                   marker='d', label=f"{algo} ({num_beams}B)", linewidth=2)
        ax.set_xlabel('Kullanıcı Sayısı', fontsize=11)
        ax.set_ylabel('Minimum SINR (dB)', fontsize=11)
        ax.set_title('Hücre Kenarı Performansı', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{output_dir}/scalability_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {filename}")
    plt.close()


def plot_sumrate_tr(df: pd.DataFrame, output_dir: str = ".", num_beams: int = None):
    """Sistem Toplam Veri Hızı vs Kullanıcı Sayısı (Türkçe) - dinamik beam sayısı"""
    # Eğer num_beams belirtilmemişse, veri setindeki ilk beam sayısını kullan
    if num_beams is None:
        available_beams = df['num_beams'].unique()
        if len(available_beams) == 0:
            print("Warning: Veri bulunamadı, sumrate grafiği atlandı")
            return
        num_beams = int(available_beams[0])
    
    df_filtered = df[df['num_beams'] == num_beams]
    if df_filtered.empty:
        print(f"Warning: num_beams == {num_beams} için sonuç bulunamadı, sumrate grafiği atlandı")
        return

    # İstediğimiz algoritma sırası
    algorithms = ['GA', 'HGA', 'PBIG', 'Max-SINR', 'Exhaustive']

    plt.figure(figsize=(8, 5))
    for algo in algorithms:
        df_algo = df_filtered[df_filtered['algorithm'] == algo].sort_values('num_ues')
        if df_algo.empty:
            continue
        plt.plot(df_algo['num_ues'], df_algo['avg_objective'],
                 marker='o', linewidth=2, label=algo)

    plt.title(f'Sistem Toplam Veri Hızı (Beams={num_beams})', fontsize=14, fontweight='bold')
    plt.xlabel('Kullanıcı Sayısı', fontsize=12)
    plt.ylabel('Sistem Toplam Veri Hızı (Mbps)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    filename = f"{output_dir}/sumrate_beams{num_beams}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {filename}")
    plt.close()


def plot_fairness_tr(df: pd.DataFrame, output_dir: str = ".", num_beams: int = None):
    """Jain adalet indeksi vs Kullanıcı Sayısı (Türkçe) - dinamik beam sayısı"""
    if 'jain_fairness' not in df.columns:
        print("Warning: jain_fairness sütunu bulunamadı, fairness grafiği atlandı")
        return

    # Eğer num_beams belirtilmemişse, veri setindeki ilk beam sayısını kullan
    if num_beams is None:
        available_beams = df['num_beams'].unique()
        if len(available_beams) == 0:
            print("Warning: Veri bulunamadı, fairness grafiği atlandı")
            return
        num_beams = int(available_beams[0])
    
    df_filtered = df[df['num_beams'] == num_beams]
    if df_filtered.empty:
        print(f"Warning: num_beams == {num_beams} için sonuç bulunamadı, fairness grafiği atlandı")
        return

    algorithms = ['GA', 'HGA', 'PBIG', 'Max-SINR', 'Exhaustive']

    plt.figure(figsize=(8, 5))
    for algo in algorithms:
        df_algo = df_filtered[df_filtered['algorithm'] == algo].sort_values('num_ues')
        if df_algo.empty:
            continue
        plt.plot(df_algo['num_ues'], df_algo['jain_fairness'],
                 marker='o', linewidth=2, label=algo)

    plt.title(f'Jain Adalet İndeksi (Beams={num_beams})', fontsize=14, fontweight='bold')
    plt.xlabel('Kullanıcı Sayısı', fontsize=12)
    plt.ylabel('Jain Adalet İndeksi', fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    filename = f"{output_dir}/fairness_beams{num_beams}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {filename}")
    plt.close()


def generate_statistics_table(df: pd.DataFrame):
    """Generate statistics table"""
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    for algo in df['algorithm'].unique():
        df_algo = df[df['algorithm'] == algo]
        print(f"\n{algo}:")
        print(f"  Objective Value:  {df_algo['avg_objective'].mean():.2f} ± {df_algo['avg_objective'].std():.2f}")
        print(f"  Runtime (ms):     {df_algo['avg_runtime_ms'].mean():.2f} ± {df_algo['avg_runtime_ms'].std():.2f}")
        print(f"  Avg SINR (dB):    {df_algo['avg_sinr_dB'].mean():.2f} ± {df_algo['avg_sinr_dB'].std():.2f}")
        print(f"  Min SINR (dB):    {df_algo['min_sinr_dB'].mean():.2f} ± {df_algo['min_sinr_dB'].std():.2f}")


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze RIC benchmark results')
    parser.add_argument('input', nargs='+', default=['ric_benchmark_results.csv'],
                       help='Input CSV file(s) with benchmark results')
    parser.add_argument('--output-dir', default='.',
                       help='Output directory for plots')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (stats only)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RIC BENCHMARK ANALYSIS")
    print("="*80)
    
    # Load results
    df = load_results(args.input)
    
    # Print statistics
    generate_statistics_table(df)
    
    # Generate plots
    if not args.no_plots:
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        try:
            plot_scalability(df, args.output_dir)
            # Veri setindeki tüm beam konfigürasyonları için grafik oluştur
            for num_beams in df['num_beams'].unique():
                plot_sumrate_tr(df, args.output_dir, int(num_beams))
                plot_fairness_tr(df, args.output_dir, int(num_beams))
            print("\n✓ All plots generated successfully!")
            print(f"\nPlots saved in: {args.output_dir}/")
            print("  - scalability_analysis.png")
            for num_beams in df['num_beams'].unique():
                print(f"  - sumrate_beams{int(num_beams)}.png")
                print(f"  - fairness_beams{int(num_beams)}.png")
        except ImportError as e:
            print(f"\nWarning: Could not generate plots: {e}")
            print("Install matplotlib: pip install matplotlib")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
