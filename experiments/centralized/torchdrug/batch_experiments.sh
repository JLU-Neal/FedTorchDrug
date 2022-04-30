rm -rf ./nohup_log/Clintox_GIN.log
python main_standalone.py --dataset=ClinTox --model=GIN >> ./nohup_log/Clintox_GIN.log 

rm -rf ./nohup_log/Clintox_GCN.log
python main_standalone.py --dataset=ClinTox --model=GCN >> ./nohup_log/Clintox_GCN.log 

rm -rf ./nohup_log/Clintox_GAT.log
python main_standalone.py --dataset=ClinTox --model=GAT >> ./nohup_log/Clintox_GAT.log 

rm -rf ./nohup_log/Clintox_NFP.log
python main_standalone.py --dataset=ClinTox --model=NFP >> ./nohup_log/Clintox_NFP.log 

rm -rf ./nohup_log/Clintox_MPNN.log
python main_standalone.py --dataset=ClinTox --model=MPNN >> ./nohup_log/Clintox_MPNN.log 