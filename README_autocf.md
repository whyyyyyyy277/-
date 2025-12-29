###train命令：python src/main.py --model_name AutoCF --dataset ML-1M --path data --emb_size 64 --lr 1e-3 --l2 1e-6 --mask_ratio 0.5 --ssl_weight 0.125 --distill_weight 0.1 --teacher_ckpt_path model/ML-1M_LightGCN_emb64.pt --epoch 200 --batch_size 1024 --gpu 0 --num_workers 0 --log_file log/ML-1M_AutoCF_emb64_mask05.txt --model_path model/ML-1M_AutoCF_emb64_mask05.pt 
###需要下载ML-1M和Grocery_and_Gourmet_Food数据集
###利用训练过的模型（上命令行例中为ML-1M_LightGCN_emb64.pt进行训练）


# AutoCF in ReChorus2.0 (HKUDS/AutoCF)

This repo adds an `AutoCF` model to ReChorus2.0 with:

- LightGCN-style graph CF backbone
- Learnable augmentation (Gumbel-sigmoid embedding masking) + contrastive SSL
- Masked graph autoencoder (MGAE) reconstruction loss (node-mask + propagation + recon)
- Frozen teacher distillation (load a teacher checkpoint and distill representations)

## Quick Start (with a ReChorus LightGCN teacher)

### 1) Train a teacher (LightGCN)

```bash
python src/main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food \
  --gpu 0 --emb_size 64 --n_layers 3 --epoch 50 --batch_size 256 --eval_batch_size 256 --num_workers 0 \
  --model_path model/LightGCN_teacher.pt --log_file log/LightGCN_teacher.txt
```

### 2) Train AutoCF with distillation + SSL

```bash
python src/main.py --model_name AutoCF --config configs/autocf.yaml \
  --teacher_ckpt_path model/LightGCN_teacher.pt \
  --log_file log/AutoCF.txt --model_path model/AutoCF.pt
```

## Notes

### Data format (ReChorus Top-K ranking)

Under `data/<dataset>/`, ReChorus expects:

- `train.csv`: `user_id`, `item_id`, `time`
- `dev.csv` / `test.csv`: `user_id`, `item_id`, `time`, `neg_items` (a Python-list string like `[1,2,3,...]`)

Evaluation uses the candidate list `[ground_truth_item] + neg_items` per row (sampled negative evaluation).
If you want full ranking, set `--test_all 1` (may be slow / high-memory on large datasets).

### Graph cache

AutoCF builds the user–item bipartite normalized adjacency from `train.csv` and caches it to:

- `data/<dataset>/autocf_norm_adj.npz` (default; configurable by `--adj_cache_name`)

Use `--regenerate 1` to rebuild caches.

### Teacher checkpoint format

`AutoCF` currently expects a ReChorus `LightGCN` checkpoint saved via `model.save_model()` (a `.pt` state_dict),
so it contains:

- `encoder.embedding_dict.user_emb`
- `encoder.embedding_dict.item_emb`

If you use an AutoCF-official teacher, convert it to this format first (or train a teacher inside ReChorus).

## Minimal smoke test

Run `scripts/smoke_autocf.py` to train a small teacher and then run AutoCF for 2–3 epochs.

```bash
python scripts/smoke_autocf.py --dataset Grocery_and_Gourmet_Food --gpu 0
```

