conda activate navdp

# activate conda env on Torch
cd /scratch/lg154/Research/Nav/InternNav                                                                                                                                                        
conda activate /scratch/lg154/conda-envs/navdp 


export PYTHONPATH="$PWD/src/diffusion-policy:${PYTHONPATH:-}"                                                                                                                                   
export CUDA_VISIBLE_DEVICES=0
                                                                                                                                                                                                  
python scripts/train/train.py \
    --name test \
    --model-name logoplanner \
    --batch-size 2 \
    --num-workers 0 \
    --epochs 1 \
    --root-dir /scratch/lg154/Research/datasets/InternData-N1/vln_n1/_raw \
    --dataset-navdp /tmp/logoplanner_dataset_lerobot.json         



cd /home/asus/Research/Nav/InternNav
export PYTHONPATH="$PWD/src/diffusion-policy:$PYTHONPATH"   # <-- required for diffusion_policy

  echo "[info] scenes to train on:"
  ls /home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i

  rm -f /tmp/navdp_smoke_dataset.json

  python scripts/train/train.py \
      --name navdp_smoke \
      --model-name navdp \
      --batch-size 2 \
      --num-workers 0 \
      --epochs 1 \
      --root-dir /home/asus/Research/datasets/InternData-N1/vln_n1/traj_data \
      --dataset-navdp /tmp/navdp_smoke_dataset.json
