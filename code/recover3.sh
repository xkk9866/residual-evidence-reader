cd /data/yaokang/proofset; L=logs/recover.log
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6
bash code/loop_run.sh hotpotqa lach fixed > logs/rec_loop2.log 2>&1; bash code/loop_eval.sh hotpotqa >> logs/rec_loop2.log 2>&1; echo "loop2 $?" >> $L
sed -i 's/^while pgrep.*//; s/^pickgpu 9000.*//; s/^pickgpu 6000.*//; s/^  \$PY code\/train3.py --term text.*//; s/RECOVER_DONE/RECOVER3_DONE/' code/recover2.sh
bash code/recover2.sh
