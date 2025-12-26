# TEZ SİMÜLASYON PARAMETRELERİ VE SİSTEM MODELİ

## TABLO 5.1: SİMÜLASYON SENARYO PARAMETRELERİ

| Parametre | Değer |
|-----------|-------|
| **Simülasyon Platformu** | ns-3 (v3.46) + 5G-LENA NR (v4.1.1) |
| **Frekans Bandı** | n78 (3.5 GHz) |
| **Bant Genişliği** | 100 MHz |
| **Çiftleme Modu** | TDD (DL:UL = 4:6) |
| **Subcarrier Spacing (SCS)** | 30 kHz (Numerology μ=1) |
| **Kanal Modeli** | 3GPP TR 38.901 Urban Micro (UMi) |
| **gNB Sayısı** | 3 (Üçgen yerleşim) |
| **gNB Arası Mesafe (ISD)** | 500 m |
| **gNB Yüksekliği** | 25 m |
| **gNB İletim Gücü** | 46 dBm |
| **gNB Gürültü Figürü** | 5 dB |
| **gNB Anten Konfigürasyonu** | UPA 4×8 = 32 eleman (Massive MIMO) |
| **Beam Sayısı (per cell)** | 8 |
| **UE Sayısı** | 10-50 (değişken) |
| **UE İletim Gücü** | 23 dBm |
| **UE Gürültü Figürü** | 9 dB |
| **UE Anten Konfigürasyonu** | UPA 2×2 = 4 eleman |
| **UE Mobility Modeli** | RandomWalk2D (300m yarıçap) |
| **Scheduler** | Proportional Fair (TDMA-PF) |
| **Trafik Modeli** | Full Buffer (UDP, 100k pps) |
| **Beamforming Yöntemi** | Direct Path Beamforming |
| **Simülasyon Süresi** | 400 ms (snapshot) |
| **SINR Toplama Periyodu** | 100 ms |

## TABLO 5.2: OPTİMİZASYON PARAMETRELERİ

| Parametre | Değer |
|-----------|-------|
| **Amaç Fonksiyonu** | F = SR × (α + (1-α) × J) |
| **Sum-Rate Hesaplama** | Shannon: R = log₂(1 + SINR) |
| **Fairness Metriği** | Jain Fairness Index |
| **Alpha (α) Değeri** | 0.7 (default) |
| **Beam Kapasitesi** | Max 4 UE/Beam |
| **İnterferans Faktörü** | 0.5 dB/UE |
| **Kaynak Paylaşımı** | Rate / beam_load |

## TABLO 5.3: ALGORİTMA PARAMETRELERİ

### Max-SINR (Baseline)
- Greedy yaklaşım: O(n)
- Beam capacity repair

### GA (Genetic Algorithm)
- Population size: 50
- Generations: 100
- Selection: Roulette Wheel
- Crossover: Single-point
- Mutation rate: 0.1

### HGA (Hybrid Genetic Algorithm)
- Population size: 50
- Generations: 100
- Local search rate: 0.3 (30% of population)
- Local search budget: 10 iterations
- + All GA operators

### PBIG (Population-Based Iterated Greedy)
- Population size: 10
- Max iterations: 100
- Destruction ratio: 0.3 (30% of UEs)
- Reconstruction: Greedy

---

## 5G-LENA KULLANAN ÖZELLİKLER

### 1. PHY KATMANI (3GPP TS 38.211-214)
- ✓ OFDM modülasyonu (NR uyumlu)
- ✓ Flexible numerology (μ=0,1,2,3,4)
- ✓ Bandwidth Part (BWP) desteği
- ✓ HARQ feedback

### 2. KANAL MODELİ (3GPP TR 38.901)
- ✓ Urban Micro (UMi) senaryo
- ✓ LOS/NLOS otomatik belirleme
- ✓ Shadowing (log-normal)
- ✓ Fast fading (CDL modeli)
- ✓ Path loss modeli

### 3. ANTEN MODELİ (3GPP TR 38.802)
- ✓ Uniform Planar Array (UPA)
- ✓ 3GPP element pattern (ThreeGppAntennaModel)
- ✓ DirectPathBeamforming
- ✓ Beam sweeping (8 beam per cell)

### 4. SINR TOPLAMA
- ✓ DlDataSinrTrace callback (PHY-layer)
- ✓ Per-beam SINR matrisi
- ✓ Real-time SINR raporlama

### 5. SCHEDULER
- ✓ Proportional Fair (NrMacSchedulerTdmaPF)
- ✓ TDMA slot allocation

---

## GERÇEK DÜNYA UYUMLULUK ANALİZİ

| Özellik | Simülasyon | Gerçek Dünya (3GPP) | Uyum |
|---------|------------|---------------------|------|
| Frekans bandı | n78 (3.5 GHz) | n78: 3.3-3.8 GHz | ✓ |
| Bant genişliği | 100 MHz | 50-100 MHz tipik | ✓ |
| gNB Tx Power | 46 dBm | 43-49 dBm | ✓ |
| Anten konfigürasyonu | 4×8 UPA | 8×8 veya 4×8 tipik | ✓ |
| Beam sayısı | 8 | 4-64 tipik | ✓ |
| Max UE per beam | 4 | PDSCH MU-MIMO: 4-8 | ✓ |
| Scheduler | Proportional Fair | Standart | ✓ |
| Kanal modeli | 3GPP TR 38.901 UMi | Standart uyumlu | ✓ |

### GENEL UYUMLULUK: ~85-90%

### Eksik/Basitleştirilmiş Özellikler
- Handover prosedürü (statik attachment)
- Inter-cell interference (sadece intra-cell)
- Dynamic TDD adaptation
- Carrier aggregation
- mmWave band desteği

---

## SİSTEM MODELİ MATEMATİKSEL FORMÜLASYON

### 1. SINR MODELİ
```
SINR_{u,b} = P_tx × G_b(θ_u) × PL(d_u)^{-1} / (N_0 × B + I)
```

Burada:
- P_tx: gNB iletim gücü (46 dBm = 39.8 W)
- G_b(θ): Beam b'nin θ açısındaki kazancı
- PL(d): Path loss (3GPP TR 38.901)
- N_0: Termal gürültü yoğunluğu (-174 dBm/Hz)
- B: Bant genişliği (100 MHz)
- I: İnterferans

### 2. RATE MODELİ (Shannon Capacity)
```
R_u = B × log_2(1 + SINR_u^{eff})
```

Effective SINR (interferans dahil):
```
SINR_u^{eff} = SINR_u - 0.5 × (n_b - 1)  [dB]
```
Burada n_b = beam b'ye atanan UE sayısı

### 3. KAYNAK PAYLAŞIMI
```
R_u^{shared} = R_u / n_b
```

### 4. JAIN FAIRNESS INDEX
```
J = (Σ R_u)² / (N × Σ R_u²)
```
J ∈ [1/N, 1], J=1 mükemmel adillik

### 5. AMAÇ FONKSİYONU
```
F = (Σ R_u) × (α + (1-α) × J)
```
α = 0.7: %70 sum-rate, %30 fairness ağırlıklı

---

## ALGORİTMALARIN ÇALIŞMA MANTIĞI

### 1. MAX-SINR (Baseline)
```python
for each UE u:
    beam_u = argmax_b(SINR_{u,b})
# + Kapasite kısıtı repair (düşük SINR'lı UE'yi taşı)
```
- Karmaşıklık: O(U × B)
- Avantaj: Çok hızlı (~5 ms)
- Dezavantaj: Global optimum değil, fairness düşük

### 2. GA (Genetic Algorithm)
```python
Chromosome: [beam_0, beam_1, ..., beam_{U-1}]

while generation < 100:
    1. Fitness hesapla (interferans dahil)
    2. Roulette Wheel Selection
    3. Single-point Crossover
    4. Random Mutation (p=0.1)
    5. Repair infeasible solutions
```
- Karmaşıklık: O(G × P × U × B)
- Avantaj: Global arama
- Dezavantaj: Convergence yavaş olabilir

### 3. HGA (Hybrid Genetic Algorithm)
```python
# GA + Local Search (Memetic)

Her nesilde:
    1. Popülasyonun %30'una local search uygula
       - 10 random neighbor dene
       - Daha iyi ise kabul et
    2. Standart GA operators
```
- Karmaşıklık: O(G × P × (1 + ls_rate × budget) × U × B)
- Avantaj: GA + local refinement = best of both worlds
- Dezavantaj: Daha yavaş

### 4. PBIG (Population-Based Iterated Greedy)
```python
# Destruction-Reconstruction paradigması

while iteration < 100:
    1. Rastgele çözüm seç
    2. Destruction: %30 UE atamasını kaldır
    3. Reconstruction: Greedy olarak yeniden ata
    4. Daha iyi ise popülasyonu güncelle
```
- Karmaşıklık: O(I × U × B²)
- Avantaj: Çeşitlilik korur, local optima'dan kaçar
- Dezavantaj: Greedy reconstruction sub-optimal olabilir

---

## SONUÇ KARŞILAŞTIRMA (V2 Benchmark)

| Algoritma | Fitness | Sum-Rate | Fairness | Runtime | Wins |
|-----------|---------|----------|----------|---------|------|
| **HGA ★** | 78.0 | 100.8 | 0.248 | 1173 ms | 6/9 |
| PBIG | 74.3 | 96.1 | 0.253 | 1139 ms | 3/9 |
| GA | 63.6 | 83.9 | 0.206 | 322 ms | 0/9 |
| Max-SINR | 38.1 | 48.1 | 0.309 ★ | 5 ms ★ | 0/9 |

### İyileştirme Oranları (vs Max-SINR)
- HGA: +105% fitness
- PBIG: +95% fitness
- GA: +67% fitness

---

## ÖZGÜN KATKILAR

1. 5G-LENA PHY-layer SINR ile metaheuristik algoritma karşılaştırması
2. İnterferans-aware fitness fonksiyonu
3. Near-RT RIC - ns-3 closed-loop entegrasyonu
4. PBIG algoritmasının beam-user assignment'a uygulanması
5. Kapsamlı alpha sensitivity ve edge-user analizi

---

## TEZ İÇİN ÖNEMLİ REFERANSLAR

### 3GPP Standartları
- 3GPP TR 38.901: Channel Model for frequencies from 0.5 to 100 GHz
- 3GPP TR 38.802: Study on New Radio (NR) Access Technology
- 3GPP TS 38.211-214: NR Physical Layer

### 5G-LENA
- CTTC 5G-LENA NR Module: https://gitlab.com/cttc-lena/nr

### O-RAN
- O-RAN Architecture Description (O-RAN.WG1.O-RAN-Architecture)
- Near-RT RIC specification (O-RAN.WG3.RICARCH)

### Metaheuristics
- Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization...
- Ruiz, R., & Stützle, T. (2007). A simple and effective iterated greedy...
