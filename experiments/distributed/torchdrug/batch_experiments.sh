# SIDER

rm -rf ./nohup_log/fed_SIDER_GCN.log
sh run_fedavg.sh 6 1 1 1 GCN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  SIDER FedAvg >> ./nohup_log/fed_SIDER_GCN.log

rm -rf ./nohup_log/fed_SIDER_GAT.log
sh run_fedavg.sh 6 1 1 1 GAT homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  SIDER FedAvg >> ./nohup_log/fed_SIDER_GAT.log

rm -rf ./nohup_log/fed_SIDER_GIN.log
sh run_fedavg.sh 6 1 1 1 GIN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  SIDER FedAvg >> ./nohup_log/fed_SIDER_GIN.log

rm -rf ./nohup_log/fed_SIDER_NFP.log
sh run_fedavg.sh 6 1 1 1 NFP homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  SIDER FedAvg >> ./nohup_log/fed_SIDER_NFP.log

rm -rf ./nohup_log/fed_SIDER_MPNN.log
# sh run_fedavg.sh 6 1 1 1 MPNN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  SIDER FedAvg >> ./nohup_log/fed_SIDER_MPNN.log



# BACE
rm -rf ./nohup_log/fed_BACE_GCN.log
sh run_fedavg.sh 6 1 1 1 GCN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BACE FedAvg >> ./nohup_log/fed_BACE_GCN.log

rm -rf ./nohup_log/fed_BACE_GAT.log
sh run_fedavg.sh 6 1 1 1 GAT homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BACE FedAvg >> ./nohup_log/fed_BACE_GAT.log

rm -rf ./nohup_log/fed_BACE_GIN.log
sh run_fedavg.sh 6 1 1 1 GIN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BACE FedAvg >> ./nohup_log/fed_BACE_GIN.log

rm -rf ./nohup_log/fed_BACE_NFP.log
sh run_fedavg.sh 6 1 1 1 NFP homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BACE FedAvg >> ./nohup_log/fed_BACE_NFP.log

rm -rf ./nohup_log/fed_BACE_MPNN.log
# sh run_fedavg.sh 6 1 1 1 MPNN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BACE FedAvg >> ./nohup_log/fed_BACE_MPNN.log


# ClinTox
rm -rf ./nohup_log/fed_ClinTox_GCN.log
sh run_fedavg.sh 6 1 1 1 GCN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  ClinTox FedAvg >> ./nohup_log/fed_ClinTox_GCN.log

rm -rf ./nohup_log/fed_ClinTox_GAT.log
sh run_fedavg.sh 6 1 1 1 GAT homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  ClinTox FedAvg >> ./nohup_log/fed_ClinTox_GAT.log

rm -rf ./nohup_log/fed_ClinTox_GIN.log
sh run_fedavg.sh 6 1 1 1 GIN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  ClinTox FedAvg >> ./nohup_log/fed_ClinTox_GIN.log

rm -rf /nohup_log/fed_ClinTox_NFP.log
sh run_fedavg.sh 6 1 1 1 NFP homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  ClinTox FedAvg >> ./nohup_log/fed_ClinTox_NFP.log

rm -rf ./nohup_log/fed_ClinTox_MPNN.log
# sh run_fedavg.sh 6 1 1 1 MPNN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  ClinTox FedAvg >> ./nohup_log/fed_ClinTox_MPNN.log


# BBBP
rm -rf ./nohup_log/fed_BBBP_GCN.log
sh run_fedavg.sh 6 1 1 1 GCN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BBBP FedAvg >> ./nohup_log/fed_BBBP_GCN.log

rm -rf ./nohup_log/fed_BBBP_GAT.log
sh run_fedavg.sh 6 1 1 1 GAT homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BBBP FedAvg >> ./nohup_log/fed_BBBP_GAT.log

rm -rf ./nohup_log/fed_BBBP_GIN.log
sh run_fedavg.sh 6 1 1 1 GIN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BBBP FedAvg >> ./nohup_log/fed_BBBP_GIN.log

rm -rf ./nohup_log/fed_BBBP_NFP.log
sh run_fedavg.sh 6 1 1 1 NFP homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BBBP FedAvg >> ./nohup_log/fed_BBBP_NFP.log

rm -rf ./nohup_log/fed_BBBP_MPNN.log
# sh run_fedavg.sh 6 1 1 1 MPNN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  BBBP FedAvg >> ./nohup_log/fed_BBBP_MPNN.log

# Tox21

rm -rf ./nohup_log/fed_Tox21_GCN.log
sh run_fedavg.sh 6 1 1 1 GCN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  Tox21 FedAvg >> ./nohup_log/fed_Tox21_GCN.log

rm -rf ./nohup_log/fed_Tox21_GAT.log
sh run_fedavg.sh 6 1 1 1 GAT homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  Tox21 FedAvg >> ./nohup_log/fed_Tox21_GAT.log

rm -rf ./nohup_log/fed_Tox21_GIN.log
sh run_fedavg.sh 6 1 1 1 GIN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  Tox21 FedAvg >> ./nohup_log/fed_Tox21_GIN.log

rm -rf ./nohup_log/fed_Tox21_NFP.log
sh run_fedavg.sh 6 1 1 1 NFP homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  Tox21 FedAvg >> ./nohup_log/fed_Tox21_NFP.log

rm -rf ./nohup_log/fed_Tox21_MPNN.log
# sh run_fedavg.sh 6 1 1 1 MPNN homo 0.5 150 1 1 0.0015 256 256 0.3 256 256  Tox21 FedAvg >> ./nohup_log/fed_Tox21_MPNN.log


