## Masked AutoEncoder for EM images

### Unique Configurations

We show below the list of configurations exclusive for MAE which extends the basic configurations in PyTorch Connectomics.

```yaml
MAE: 
  MODEL: 'vit-base' # MAE encoder, includes 'vit-base', 'vit-large', and 'vit-huge'.
  MASKRATIO: 0.9 # Mask ratio
  PRETRAIN: True # For the fine-tuning process.  
```

### Command

MAE pre-training command using distributed data parallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -u -m torch.distributed.run \
--nproc_per_node=2 --master_port=9967 projects/EmMAE/main.py --distributed \
--config-file projects/EmMAE/config/SNEMI/MAE-Base.yaml 
```

Fine-tuning command using distributed data parallel on `SNEMI3D` with `Affinity map`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python -u -m torch.distributed.run \
--nproc_per_node=2 --master_port=9967 projects/EmMAE/ftmain.py --distributed \
--config-base projects/EmMAE/config/SNEMI/FT-VitBase.yaml \
--config-file projects/EmMAE/config/SNEMI/FT-VitBase-Affinity.yaml
```

Inference command using data parallel:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u projects/EmMAE/main.py \
--inference --config-base projects/EmMAE/config/SNEMI/FT-VitBase.yaml \
--config-file projects/EmMAE/config/SNEMI/FT-VitBase-Affinity.yaml \
--checkpoint path/to/your/checkpoint
```
