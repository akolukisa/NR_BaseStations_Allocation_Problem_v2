#!/bin/bash
# 4 Beam için farklı UE sayılarında LENA simülasyonu

NS3_DIR="/Users/akolukisa/workspace/ns-3-dev"
OUTPUT_DIR="/Users/akolukisa/FinalThesis/ric-python/lena_scalability_4beam"

mkdir -p $OUTPUT_DIR

# UE sayıları ve gNB başına UE
declare -A UE_MAP
UE_MAP[10]=4    # 4*3=12 ≈ 10
UE_MAP[15]=5    # 5*3=15
UE_MAP[20]=7    # 7*3=21 ≈ 20
UE_MAP[25]=9    # 9*3=27 ≈ 25
UE_MAP[30]=10   # 10*3=30
UE_MAP[35]=12   # 12*3=36 ≈ 35
UE_MAP[40]=14   # 14*3=42 ≈ 40
UE_MAP[45]=15   # 15*3=45
UE_MAP[50]=17   # 17*3=51 ≈ 50

cd $NS3_DIR

for TARGET_UE in 10 15 20 25 30 35 40 45 50; do
    UE_PER_GNB=${UE_MAP[$TARGET_UE]}
    echo "=========================================="
    echo "Running: $TARGET_UE UE (${UE_PER_GNB} per gNB), 4 beams"
    echo "=========================================="
    
    # Simülasyonu çalıştır
    ./ns3 run "thesis-nr-scenario --numBeams=4 --ueNumPergNb=${UE_PER_GNB}" 2>&1 | tail -20
    
    # Sonuçları kaydet
    DEST_DIR="${OUTPUT_DIR}/ue${TARGET_UE}"
    mkdir -p $DEST_DIR
    
    if [ -d "thesis-results" ]; then
        cp thesis-results/sinr_gnb*.json $DEST_DIR/ 2>/dev/null
        echo "SINR files copied to $DEST_DIR"
        ls -la $DEST_DIR/
    fi
    
    echo ""
done

echo "Tüm simülasyonlar tamamlandı!"
echo "Sonuçlar: $OUTPUT_DIR"
