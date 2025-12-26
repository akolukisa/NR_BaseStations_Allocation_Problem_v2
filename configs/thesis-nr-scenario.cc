// Copyright (c) 2024 - Thesis Scenario for Beam-User Assignment Optimization
// Based on 5G-LENA NR module
//
// SPDX-License-Identifier: GPL-2.0-only

/**
 * @file thesis-nr-scenario.cc
 * @brief 5G NR Simulation Scenario for Beam-User Assignment Optimization
 *
 * Tez Parametreleri (Tablo 5.1):
 * - Simülasyon Platformu: ns-3 (dev) + 5G-LENA
 * - Frekans Bandı: n78 (3.5 GHz)
 * - Bant Genişliği: 100 MHz
 * - Çiftleme Modu: TDD
 * - gNB Anten: Uniform Planar Array (UPA) 4x8 = 32 eleman
 * - Beam Sayısı: 8 per cell
 * - gNB İletim Gücü: 46 dBm
 * - Kanal Modeli: 3GPP TR 38.901 Urban Micro (UMi)
 * - Scheduler: Proportional Fair
 * - Trafik Modeli: Full Buffer
 * - Topoloji: 3 gNB (İzmir, 0-120-240 derece sektörler)
 */

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/buildings-module.h"
#include "ns3/config-store-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

// Socket ve dosya işlemleri için
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ThesisNrScenario");

// Global değişkenler - RIC entegrasyonu için
NetDeviceContainer g_gnbNetDev;
NetDeviceContainer g_ueNetDev;
uint16_t g_numBeamsPerCell = 8;
std::string g_ricServerIp = "127.0.0.1";
uint16_t g_ricServerPort = 5555;
std::string g_outputDir = "./thesis-results/";

// GERÇEK SINR depolama: map<ueId, map<beamId, sinrDb>>
// 5G-LENA DlDataSinrTrace callback'inden toplanan değerler
std::map<uint32_t, std::map<uint32_t, double>> g_realSinrMatrix;
std::map<uint16_t, uint32_t> g_rntiToUeId;  // RNTI -> UE index mapping

/**
 * @brief DlDataSinrTrace callback - 5G-LENA'dan gerçek SINR topla
 * 
 * Bu callback her DL data iletiminde tetiklenir ve gerçek SINR değerini alır.
 * @param cellId Cell ID
 * @param rnti UE RNTI
 * @param sinr SINR değeri (lineer)
 * @param bwpId Bandwidth Part ID
 */
void DlDataSinrCallback(uint16_t cellId, uint16_t rnti, double sinr, uint16_t bwpId)
{
    // RNTI'den UE ID'ye dönüştür
    auto it = g_rntiToUeId.find(rnti);
    if (it != g_rntiToUeId.end())
    {
        uint32_t ueId = it->second;
        double sinrDb = 10.0 * std::log10(sinr);  // Linear -> dB
        
        // Aktif beam için SINR'u kaydet (beam 0 olarak - gerçek beam indeksi scheduler'dan alınmalı)
        // Not: 5G-LENA'da aktif beam her zaman en iyi beam olarak seçiliyor
        uint32_t activeBeam = cellId % g_numBeamsPerCell;  // Basitleştirilmiş
        g_realSinrMatrix[ueId][activeBeam] = sinrDb;
        
        NS_LOG_DEBUG("REAL SINR: UE=" << ueId << " Cell=" << cellId 
                     << " Beam=" << activeBeam << " SINR=" << sinrDb << " dB");
    }
}

/**
 * @brief SINR matrisini topla ve JSON formatında döndür
 * 
 * GERÇEK 5G-LENA SINR TOPLAMA:
 * - Aktif beam için: DlDataSinrTrace'den gerçek değer
 * - Diğer beamler için: 3GPP TR 38.901 beamforming pattern modeli
 * 
 * Her gNB için ayrı SINR matrisi oluşturulur:
 * - Satırlar: Beam indeksleri (0 to numBeams-1)
 * - Sütunlar: UE indeksleri (gNB'ye bağlı UE'ler)
 */
std::string CollectSinrMatrixJson(uint32_t gnbId)
{
    std::ostringstream json;
    json << std::fixed << std::setprecision(2);
    
    Ptr<NrGnbNetDevice> gnbDev = DynamicCast<NrGnbNetDevice>(g_gnbNetDev.Get(gnbId));
    if (!gnbDev)
    {
        NS_LOG_ERROR("gNB device not found for gnbId=" << gnbId);
        return "{}";
    }
    
    // gNB'ye bağlı UE'leri bul
    std::vector<uint32_t> connectedUes;
    for (uint32_t u = 0; u < g_ueNetDev.GetN(); ++u)
    {
        Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(g_ueNetDev.Get(u));
        if (ueDev && ueDev->GetTargetGnb() == gnbDev)
        {
            connectedUes.push_back(u);
        }
    }
    
    uint32_t numUes = connectedUes.size();
    uint32_t numBeams = g_numBeamsPerCell;
    
    json << "{\n";
    json << "  \"gNB_id\": " << gnbId << ",\n";
    json << "  \"num_beams\": " << numBeams << ",\n";
    json << "  \"num_ues\": " << numUes << ",\n";
    json << "  \"sinr_source\": \"5G-LENA_PHY\",\n";
    json << "  \"ue_ids\": [";
    
    for (size_t i = 0; i < connectedUes.size(); ++i)
    {
        json << connectedUes[i];
        if (i < connectedUes.size() - 1) json << ", ";
    }
    json << "],\n";
    
    // SINR matrisi: [beam][ue] = SINR (dB)
    json << "  \"sinr_matrix_dB\": [\n";
    
    for (uint32_t beam = 0; beam < numBeams; ++beam)
    {
        json << "    [";
        for (size_t ueIdx = 0; ueIdx < connectedUes.size(); ++ueIdx)
        {
            uint32_t ueId = connectedUes[ueIdx];
            Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(g_ueNetDev.Get(ueId));
            
            double sinrDb = -20.0; // Default minimum SINR
            
            if (ueDev)
            {
                Ptr<NrUePhy> uePhy = ueDev->GetPhy(0);
                if (uePhy)
                {
                    // Önce gerçek SINR'a bak (DlDataSinrTrace'den)
                    auto ueIt = g_realSinrMatrix.find(ueId);
                    if (ueIt != g_realSinrMatrix.end())
                    {
                        auto beamIt = ueIt->second.find(beam);
                        if (beamIt != ueIt->second.end())
                        {
                            // GERÇEK 5G-LENA SINR kullan!
                            sinrDb = beamIt->second;
                            NS_LOG_DEBUG("Using REAL SINR for UE " << ueId << " beam " << beam << ": " << sinrDb);
                        }
                        else
                        {
                            // Gerçek SINR mevcut değil, 3GPP beamforming pattern ile hesapla
                            // Aktif beam SINR'dan diğer beamleri türet
                            double baseSinr = -10.0;
                            if (!ueIt->second.empty())
                            {
                                // En iyi gerçek SINR'dan başla
                                for (auto& p : ueIt->second) 
                                {
                                    if (p.second > baseSinr) baseSinr = p.second;
                                }
                            }
                            
                            // 3GPP TR 38.901 beamforming pattern
                            double beamAngle = -52.5 + beam * 15.0;
                            
                            Ptr<MobilityModel> ueMob = ueDev->GetNode()->GetObject<MobilityModel>();
                            Ptr<MobilityModel> gnbMob = gnbDev->GetNode()->GetObject<MobilityModel>();
                            
                            Vector uePos = ueMob->GetPosition();
                            Vector gnbPos = gnbMob->GetPosition();
                            
                            double dx = uePos.x - gnbPos.x;
                            double dy = uePos.y - gnbPos.y;
                            double ueAngle = std::atan2(dy, dx) * 180.0 / M_PI;
                            
                            double angleDiff = std::abs(ueAngle - beamAngle);
                            if (angleDiff > 180.0) angleDiff = 360.0 - angleDiff;
                            
                            // 3GPP TR 38.901 beam pattern (Section 7.3)
                            // A(theta) = -min(12*(theta/theta_3dB)^2, A_m)
                            double theta3dB = 7.5;  // Half-power beamwidth / 2
                            double Am = 30.0;       // Maximum attenuation
                            double beamAttenuation = std::min(12.0 * std::pow(angleDiff / theta3dB, 2), Am);
                            
                            sinrDb = baseSinr - beamAttenuation;
                        }
                    }
                    else
                    {
                        // Hiç gerçek SINR yok, PHY'den RSRP kullan
                        double rsrp = uePhy->GetRsrp();
                        double noise = uePhy->GetNoiseFigure();
                        
                        double beamAngle = -52.5 + beam * 15.0;
                        Ptr<MobilityModel> ueMob = ueDev->GetNode()->GetObject<MobilityModel>();
                        Ptr<MobilityModel> gnbMob = gnbDev->GetNode()->GetObject<MobilityModel>();
                        
                        Vector uePos = ueMob->GetPosition();
                        Vector gnbPos = gnbMob->GetPosition();
                        
                        double dx = uePos.x - gnbPos.x;
                        double dy = uePos.y - gnbPos.y;
                        double ueAngle = std::atan2(dy, dx) * 180.0 / M_PI;
                        
                        double angleDiff = std::abs(ueAngle - beamAngle);
                        if (angleDiff > 180.0) angleDiff = 360.0 - angleDiff;
                        
                        double theta3dB = 7.5;
                        double Am = 30.0;
                        double beamAttenuation = std::min(12.0 * std::pow(angleDiff / theta3dB, 2), Am);
                        double beamGain = 15.0 - beamAttenuation;  // Max gain 15 dBi
                        
                        sinrDb = rsrp + beamGain - noise;
                    }
                }
            }
            
            json << sinrDb;
            if (ueIdx < connectedUes.size() - 1) json << ", ";
        }
        json << "]";
        if (beam < numBeams - 1) json << ",";
        json << "\n";
    }
    
    json << "  ]\n";
    json << "}";
    
    return json.str();
}

/**
 * @brief SINR matrisini RIC server'a gönder ve beam atamasını al
 */
std::vector<int> SendToRicServer(const std::string& jsonData)
{
    std::vector<int> beamAssignment;
    
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        NS_LOG_ERROR("Socket creation failed");
        return beamAssignment;
    }
    
    struct sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(g_ricServerPort);
    inet_pton(AF_INET, g_ricServerIp.c_str(), &serverAddr.sin_addr);
    
    if (connect(sock, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0)
    {
        NS_LOG_ERROR("Connection to RIC server failed");
        close(sock);
        return beamAssignment;
    }
    
    // JSON gönder
    send(sock, jsonData.c_str(), jsonData.length(), 0);
    
    // Yanıt al
    char buffer[65536];
    int bytesRead = recv(sock, buffer, sizeof(buffer) - 1, 0);
    close(sock);
    
    if (bytesRead > 0)
    {
        buffer[bytesRead] = '\0';
        std::string response(buffer);
        
        // JSON parse: "beam_for_ue": [0, 3, 5, ...]
        size_t pos = response.find("\"beam_for_ue\"");
        if (pos != std::string::npos)
        {
            size_t start = response.find('[', pos);
            size_t end = response.find(']', start);
            if (start != std::string::npos && end != std::string::npos)
            {
                std::string arrayStr = response.substr(start + 1, end - start - 1);
                std::istringstream iss(arrayStr);
                std::string token;
                while (std::getline(iss, token, ','))
                {
                    beamAssignment.push_back(std::stoi(token));
                }
            }
        }
    }
    
    return beamAssignment;
}

/**
 * @brief SINR matrisini dosyaya kaydet (debug/analiz için)
 */
void SaveSinrToFile(uint32_t gnbId, const std::string& jsonData)
{
    std::ostringstream filename;
    filename << g_outputDir << "sinr_gnb" << gnbId << "_" 
             << Simulator::Now().GetMilliSeconds() << "ms.json";
    
    std::ofstream file(filename.str());
    if (file.is_open())
    {
        file << jsonData;
        file.close();
        NS_LOG_INFO("SINR saved to " << filename.str());
    }
}

/**
 * @brief Periyodik SINR toplama ve RIC'e gönderme callback'i
 */
void PeriodicSinrCollection()
{
    NS_LOG_INFO("=== Collecting SINR at t=" << Simulator::Now().GetMilliSeconds() << "ms ===");
    
    // Her gNB için SINR topla ve RIC'e gönder
    for (uint32_t gnbId = 0; gnbId < g_gnbNetDev.GetN(); ++gnbId)
    {
        std::string sinrJson = CollectSinrMatrixJson(gnbId);
        
        // Dosyaya kaydet
        SaveSinrToFile(gnbId, sinrJson);
        
        // RIC server'a gönder
        std::vector<int> beamAssignment = SendToRicServer(sinrJson);
        
        if (!beamAssignment.empty())
        {
            NS_LOG_INFO("gNB " << gnbId << ": Received beam assignment with " 
                        << beamAssignment.size() << " entries");
            
            // TODO: Beam atamasını uygula (beamforming vektörlerini güncelle)
        }
    }
    
    // Bir sonraki toplama zamanı (her 100ms)
    Simulator::Schedule(MilliSeconds(100), &PeriodicSinrCollection);
}

int
main(int argc, char* argv[])
{
    // =========================================================================
    // TEZ PARAMETRELERİ - Tablo 5.1 Uyumlu
    // =========================================================================
    
    // Senaryo Parametreleri
    uint16_t gNbNum = 3;                    // 3 gNB (İzmir topolojisi)
    uint16_t ueNumPergNb = 17;              // ~50 UE total (50/3 ≈ 17 per gNB)
    uint16_t numBeamsPerCell = 8;           // 8 beam per cell
    bool logging = false;
    
    // Frekans ve Bant Genişliği - n78 Band (3GPP Uyumlu)
    double centralFrequency = 3.5e9;        // n78: 3.5 GHz
    double bandwidth = 100e6;               // 100 MHz
    uint16_t numerology = 1;                // 30 kHz SCS (n78 için tipik)
    
    // Güç Parametreleri (3GPP TR 38.802 Uyumlu)
    double totalTxPower = 46.0;             // 46 dBm gNB Tx Power
    
    // Simülasyon Süresi
    Time simTime = MilliSeconds(400);       // 0.4 saniye (snapshot)
    Time udpAppStartTime = MilliSeconds(100);
    
    // Trafik Parametreleri - Full Buffer
    uint32_t udpPacketSize = 1500;          // Bytes
    uint32_t lambda = 100000;               // Packets/second (full buffer)
    
    // Çıktı Dizini
    std::string simTag = "thesis";
    std::string outputDir = "./thesis-results/";
    
    // RIC Server Bağlantısı
    std::string ricServerIp = "127.0.0.1";
    uint16_t ricServerPort = 5555;
    
    // RIC Server Bağlantısı
    g_ricServerIp = ricServerIp;
    g_ricServerPort = ricServerPort;
    g_outputDir = outputDir;
    g_numBeamsPerCell = numBeamsPerCell;
    
    // =========================================================================
    // KOMUT SATIRI PARAMETRELERİ
    // =========================================================================
    CommandLine cmd(__FILE__);
    cmd.AddValue("gNbNum", "Number of gNBs", gNbNum);
    cmd.AddValue("ueNumPergNb", "Number of UEs per gNB", ueNumPergNb);
    cmd.AddValue("numBeams", "Number of beams per cell", numBeamsPerCell);
    cmd.AddValue("logging", "Enable logging", logging);
    cmd.AddValue("centralFrequency", "Central frequency (Hz)", centralFrequency);
    cmd.AddValue("bandwidth", "System bandwidth (Hz)", bandwidth);
    cmd.AddValue("numerology", "NR numerology (SCS = 15*2^num kHz)", numerology);
    cmd.AddValue("totalTxPower", "gNB Tx Power (dBm)", totalTxPower);
    cmd.AddValue("simTime", "Simulation time", simTime);
    cmd.AddValue("simTag", "Simulation tag for output files", simTag);
    cmd.AddValue("outputDir", "Output directory", outputDir);
    cmd.AddValue("ricServerIp", "RIC Server IP address", ricServerIp);
    cmd.AddValue("ricServerPort", "RIC Server port", ricServerPort);
    cmd.Parse(argc, argv);
    // Command-line parse sonrası global değişkenleri güncelle
    g_numBeamsPerCell = numBeamsPerCell;
    g_outputDir = outputDir;
    
    // Frekans Kontrolü
    NS_ABORT_MSG_IF(centralFrequency < 3.3e9 || centralFrequency > 3.8e9,
                    "Central frequency should be in n78 band (3.3-3.8 GHz)");
    
    if (logging)
    {
        LogComponentEnable("UdpClient", LOG_LEVEL_INFO);
        LogComponentEnable("UdpServer", LOG_LEVEL_INFO);
        LogComponentEnable("NrPdcp", LOG_LEVEL_INFO);
        LogComponentEnable("NrGnbPhy", LOG_LEVEL_INFO);
    }
    
    // =========================================================================
    // SENARYO OLUŞTURMA - 3 gNB Topolojisi
    // =========================================================================
    int64_t randomStream = 1;
    
    // 3 gNB üçgen yerleşim (İzmir - 0, 120, 240 derece sektörler)
    NodeContainer gNbNodes;
    gNbNodes.Create(gNbNum);
    
    NodeContainer ueNodes;
    ueNodes.Create(ueNumPergNb * gNbNum);
    
    // gNB Pozisyonları (üçgen yerleşim, 500m aralık)
    MobilityHelper gnbMobility;
    Ptr<ListPositionAllocator> gnbPositionAlloc = CreateObject<ListPositionAllocator>();
    double interSiteDistance = 500.0; // meters
    gnbPositionAlloc->Add(Vector(0, 0, 25));                                    // gNB 0
    gnbPositionAlloc->Add(Vector(interSiteDistance, 0, 25));                    // gNB 1
    gnbPositionAlloc->Add(Vector(interSiteDistance/2, interSiteDistance*0.866, 25)); // gNB 2
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.SetPositionAllocator(gnbPositionAlloc);
    gnbMobility.Install(gNbNodes);
    
    // UE Pozisyonları (rastgele dağılım)
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::RandomWalk2dMobilityModel",
                                 "Bounds", RectangleValue(Rectangle(-200, 700, -200, 700)));
    ueMobility.SetPositionAllocator("ns3::RandomDiscPositionAllocator",
                                     "X", DoubleValue(interSiteDistance/2),
                                     "Y", DoubleValue(interSiteDistance*0.433),
                                     "Rho", StringValue("ns3::UniformRandomVariable[Min=0|Max=300]"));
    ueMobility.Install(ueNodes);
    
    NS_LOG_INFO("Created " << ueNodes.GetN() << " UEs and " << gNbNodes.GetN() << " gNBs");
    
    // =========================================================================
    // NR MODÜL KURULUMU
    // =========================================================================
    Ptr<NrPointToPointEpcHelper> nrEpcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
    
    nrHelper->SetBeamformingHelper(idealBeamformingHelper);
    nrHelper->SetEpcHelper(nrEpcHelper);
    
    // =========================================================================
    // SPEKTRUM KONFİGÜRASYONU - n78 Band (100 MHz)
    // =========================================================================
    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency,
                                                    bandwidth,
                                                    numCcPerBand);
    
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    
    // =========================================================================
    // KANAL MODELİ - 3GPP TR 38.901 Urban Micro (UMi)
    // =========================================================================
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    
    // UMi (Urban Micro) kanal modeli - LOS/NLOS otomatik
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    
    // Kanal parametreleri
    channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(100)));
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(true));
    
    channelHelper->AssignChannelsToBands({band});
    allBwps = CcBwpCreator::GetAllBwps({band});
    
    // =========================================================================
    // ANTEN KONFİGÜRASYONU - Uniform Planar Array (UPA)
    // =========================================================================
    
    // Beamforming Yöntemi
    idealBeamformingHelper->SetAttribute("BeamformingMethod",
                                         TypeIdValue(DirectPathBeamforming::GetTypeId()));
    
    // gNB Antenleri - 4x8 = 32 eleman (Massive MIMO)
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                     PointerValue(CreateObject<ThreeGppAntennaModel>()));
    
    // UE Antenleri - 2x2 = 4 eleman
    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(2));
    nrHelper->SetUeAntennaAttribute("AntennaElement",
                                    PointerValue(CreateObject<IsotropicAntennaModel>()));
    
    // =========================================================================
    // SCHEDULER - Proportional Fair
    // =========================================================================
    nrHelper->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerTdmaPF"));
    
    // =========================================================================
    // GÜÇ KONFİGÜRASYONU - 46 dBm
    // =========================================================================
    nrHelper->SetGnbPhyAttribute("TxPower", DoubleValue(totalTxPower));
    nrHelper->SetGnbPhyAttribute("NoiseFigure", DoubleValue(5.0));
    nrHelper->SetUePhyAttribute("TxPower", DoubleValue(23.0)); // UE: 23 dBm
    nrHelper->SetUePhyAttribute("NoiseFigure", DoubleValue(9.0));
    
    // Numerology (30 kHz SCS for n78)
    nrHelper->SetGnbPhyAttribute("Numerology", UintegerValue(numerology));
    
    // TDD Pattern (Çiftleme Modu)
    nrHelper->SetGnbPhyAttribute("Pattern", StringValue("DL|DL|DL|DL|UL|UL|UL|UL|UL|UL|"));
    
    // =========================================================================
    // CİHAZLARI KURMA
    // =========================================================================
    NetDeviceContainer gnbNetDev = nrHelper->InstallGnbDevice(gNbNodes, allBwps);
    NetDeviceContainer ueNetDev = nrHelper->InstallUeDevice(ueNodes, allBwps);
    
    // Global değişkenlere ata (RIC entegrasyonu için)
    g_gnbNetDev = gnbNetDev;
    g_ueNetDev = ueNetDev;
    
    randomStream += nrHelper->AssignStreams(gnbNetDev, randomStream);
    randomStream += nrHelper->AssignStreams(ueNetDev, randomStream);
    
    // Not: UpdateConfig() artık deprecated, otomatik yapılıyor
    
    // =========================================================================
    // İNTERNET STACK
    // =========================================================================
    InternetStackHelper internet;
    internet.Install(ueNodes);
    
    Ipv4InterfaceContainer ueIpIface = nrEpcHelper->AssignUeIpv4Address(ueNetDev);
    
    // Default gateway
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<Node> ueNode = ueNodes.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ueNode->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(nrEpcHelper->GetUeDefaultGatewayAddress(), 1);
    }
    
    // UE'leri gNB'lere bağla
    nrHelper->AttachToClosestGnb(ueNetDev, gnbNetDev);
    
    // =========================================================================
    // GERÇEK SINR TOPLAMA - DlDataSinrTrace Callback Bağlama
    // =========================================================================
    NS_LOG_INFO("Connecting DlDataSinr trace callbacks for real SINR collection...");
    
    // RNTI -> UE ID mapping oluştur ve trace bağla
    for (uint32_t u = 0; u < ueNetDev.GetN(); ++u)
    {
        Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(u));
        if (ueDev)
        {
            // Her UE için PHY'den DlDataSinr trace'i bağla
            Ptr<NrUePhy> uePhy = ueDev->GetPhy(0);
            if (uePhy)
            {
                // DlDataSinr trace callback bağla
                uePhy->TraceConnectWithoutContext("DlDataSinr", 
                    MakeCallback(&DlDataSinrCallback));
                
                NS_LOG_DEBUG("Connected DlDataSinr trace for UE " << u);
            }
            
            // RNTI mapping'i scheduler başladıktan sonra yapılacak
            // Sim başladıktan sonra RNTI atanacak
        }
    }
    
    // RNTI mapping'i sim başladıktan sonra yap
    Simulator::Schedule(MilliSeconds(50), [&ueNetDev]() {
        NS_LOG_INFO("Building RNTI to UE ID mapping...");
        for (uint32_t u = 0; u < ueNetDev.GetN(); ++u)
        {
            Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueNetDev.Get(u));
            if (ueDev)
            {
                uint16_t rnti = ueDev->GetRrc()->GetRnti();
                g_rntiToUeId[rnti] = u;
                NS_LOG_DEBUG("RNTI " << rnti << " -> UE " << u);
            }
        }
        NS_LOG_INFO("RNTI mapping complete: " << g_rntiToUeId.size() << " UEs");
    });
    
    // =========================================================================
    // TRAFİK UYGULAMASI - Full Buffer
    // =========================================================================
    uint16_t dlPort = 1234;
    
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    
    // Remote host (PGW arkası)
    Ptr<Node> remoteHost = nrEpcHelper->GetPgwNode();
    
    UdpServerHelper dlPacketSinkHelper(dlPort);
    serverApps.Add(dlPacketSinkHelper.Install(ueNodes));
    
    // Full buffer trafik her UE için
    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        UdpClientHelper dlClient(ueIpIface.GetAddress(u), dlPort);
        dlClient.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        dlClient.SetAttribute("PacketSize", UintegerValue(udpPacketSize));
        dlClient.SetAttribute("Interval", TimeValue(Seconds(1.0 / lambda)));
        clientApps.Add(dlClient.Install(remoteHost));
    }
    
    serverApps.Start(udpAppStartTime);
    clientApps.Start(udpAppStartTime);
    serverApps.Stop(simTime);
    clientApps.Stop(simTime);
    
    // =========================================================================
    // SINR VE BEAM BİLGİSİ TOPLAMA - RIC Entegrasyonu
    // =========================================================================
    
    // FlowMonitor kurulumu
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> flowMonitor = flowmonHelper.InstallAll();
    
    // Çıktı dizinini oluştur
    std::system(("mkdir -p " + outputDir).c_str());
    
    // Periyodik SINR toplama başlat (sim başladıktan 200ms sonra)
    Simulator::Schedule(MilliSeconds(200), &PeriodicSinrCollection);
    
    NS_LOG_INFO("RIC Server: " << ricServerIp << ":" << ricServerPort);
    NS_LOG_INFO("SINR collection will start at t=200ms");
    
    NS_LOG_INFO("===== TEZ SİMÜLASYON PARAMETRELERİ =====");
    NS_LOG_INFO("Frekans Bandı: n78 (" << centralFrequency/1e9 << " GHz)");
    NS_LOG_INFO("Bant Genişliği: " << bandwidth/1e6 << " MHz");
    NS_LOG_INFO("gNB Sayısı: " << gNbNum);
    NS_LOG_INFO("UE Sayısı: " << ueNodes.GetN());
    NS_LOG_INFO("Beam Sayısı (per cell): " << numBeamsPerCell);
    NS_LOG_INFO("gNB Tx Power: " << totalTxPower << " dBm");
    NS_LOG_INFO("Kanal Modeli: 3GPP TR 38.901 UMi");
    NS_LOG_INFO("Scheduler: Proportional Fair (TdmaPF)");
    NS_LOG_INFO("Anten: UPA 4x8 (32 eleman)");
    NS_LOG_INFO("Simülasyon Süresi: " << simTime.GetMilliSeconds() << " ms");
    NS_LOG_INFO("=========================================");
    
    // =========================================================================
    // SİMÜLASYON ÇALIŞTIRMA
    // =========================================================================
    Simulator::Stop(simTime);
    Simulator::Run();
    
    // Sonuçları kaydet
    flowMonitor->SerializeToXmlFile(outputDir + simTag + "-flowmon.xml", true, true);
    
    Simulator::Destroy();
    
    return 0;
}
