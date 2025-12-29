# AutoCF Reproduction in ReChorus Framework

This repository contains the reproduction and ablation studies of the **AutoCF (Automated Collaborative Filtering)** model, implemented within the [ReChorus](https://github.com/THUwangcy/ReChorus) framework.

The implementation focuses on verifying the effectiveness of **Masked Graph Autoencoder (MGAE)**, **Contrastive Learning (SSL)**, and **Knowledge Distillation** in recommendation tasks.

## 🚀 Features

- **End-to-End Implementation**: Complete pipeline from data preprocessing, Teacher model training, to AutoCF Student training.
- **Ablation Studies**: Pre-configured scripts to verify the contribution of each module (Distillation, SSL, Masking).
- **Dual Dataset Support**: Tested on both dense (**ML-1M**) and sparse (**Grocery_and_Gourmet_Food**) datasets.

## 📂 Project Structure

```text
├── data/                  # Dataset directory
├── log/                   # Training logs and results
├── model/                 # Saved model checkpoints (.pt)
├── src/
│   ├── models/general/
│   │   └── AutoCF.py      # Core AutoCF Model Implementation
│   ├── helpers/
│   │   └── AutoCFRunner.py # Specialized Runner for AutoCF
│   └── main.py            # Entry point
└── README.md
🛠️ Requirements
Python 3.8+
PyTorch 1.12+ (Tested on 1.13 with CUDA 11.7)
NumPy, Pandas, Scipy
ReChorus dependencies
Install dependencies via:
code
Bash
pip install -r requirements.txt
📊 Data Preparation
We use the standard ReChorus data format. You can use the provided script to generate the 70/5/25 split.
Example: Processing ML-1M
code
Bash
# Ensure you have the raw ratings.dat in data/ML-1M/
python scripts/prepare_autocf_data.py --input data/ML-1M/ratings.dat --dataset ML-1M --out_dir data --split 0.7,0.05,0.25 --eval_neg 99 --sep "::" --user_col 0 --item_col 1 --time_col 3
🏃‍♂️ Training Guide
Training AutoCF involves two steps: Teacher Training and Student Training.
Step 1: Train Teacher (LightGCN)
AutoCF requires a pre-trained LightGCN as the teacher.
code
Bash
python src/main.py --model_name LightGCN --dataset ML-1M --path data --emb_size 64 --lr 1e-3 --epoch 200 --batch_size 1024 --num_workers 4 --log_file log/ML-1M_Teacher.txt --model_path model/ML-1M_LightGCN_emb64.pt
Step 2: Train AutoCF (Student)
Run the full AutoCF model with Distillation and SSL enabled.
code
Bash
python src/main.py --model_name AutoCF --dataset ML-1M --path data --emb_size 64 --lr 1e-3 --l2 1e-6 --mask_ratio 0.1 --ssl_weight 0.125 --distill_weight 0.1 --teacher_ckpt_path model/ML-1M_LightGCN_emb64.pt --epoch 200 --batch_size 1024 --gpu 0 --num_workers 0 --log_file log/ML-1M_AutoCF_Full.txt --model_path model/ML-1M_AutoCF_Full.pt 
🧪 Ablation Studies
We conducted ablation studies to analyze the impact of Distillation, SSL, and Masking.
Variant	Command Argument Changes	Description
Full Model	(Default)	All components enabled.
w/o Distill	--distill_weight 0	Removes Knowledge Distillation from Teacher.
w/o SSL	--ssl_weight 0	Removes Contrastive Learning loss.
w/o Mask	--mask_ratio 0	Removes MGAE masking mechanism.
Commands for Ablation
1. Without Distillation:
code
Bash
python src/main.py --model_name AutoCF --dataset ML-1M --path data --emb_size 64 --lr 1e-3 --l2 1e-6 --mask_ratio 0.1 --ssl_weight 0.125 --distill_weight 0 --teacher_ckpt_path model/ML-1M_LightGCN_emb64.pt --epoch 200 --batch_size 1024 --gpu 0 --num_workers 0 --log_file log/ML-1M_AutoCF_emb64_noDISTILL.txt --model_path model/ML-1M_AutoCF_emb64_noDISTILL.pt 
2. Without SSL:
code
Bash
python src/main.py --model_name AutoCF --dataset ML-1M --path data --emb_size 64 --lr 1e-3 --l2 1e-6 --mask_ratio 0.1 --ssl_weight 0 --distill_weight 0.1 --teacher_ckpt_path model/ML-1M_LightGCN_emb64.pt --epoch 200 --batch_size 1024 --gpu 0 --num_workers 0 --log_file log/ML-1M_AutoCF_emb64_noSSL.txt --model_path model/ML-1M_AutoCF_emb64_noSSL.pt 
3. Without Masking (MGAE):
code
Bash
python src/main.py --model_name AutoCF --dataset ML-1M --path data --emb_size 64 --lr 1e-3 --l2 1e-6 --mask_ratio 0 --ssl_weight 0.125 --distill_weight 0.1 --teacher_ckpt_path model/ML-1M_LightGCN_emb64.pt --epoch 200 --batch_size 1024 --gpu 0 --num_workers 0 --log_file log/ML-1M_AutoCF_emb64_noMGAE.txt --model_path model/ML-1M_AutoCF_emb64_noMGAE.pt 
## 📊 Experimental Results

We evaluated the performance of AutoCF against baselines (NeuMF, LightGCN) on both sparse and dense datasets. We also analyzed the impact of the SSL weight parameter.

### Table 1: Performance on Grocery_and_Gourmet_Food (Sparse Dataset)

| Model | HR@5 | NDCG@5 | HR@20 | Improv. vs LightGCN (NDCG@5) |
| :--- | :---: | :---: | :---: | :---: |
| NeuMF | 0.2904 | 0.1955 | 0.5261 | -27.2% |
| LightGCN | 0.3891 | 0.2683 | 0.6303 | - |
| **AutoCF (ssl=0.01)** | **0.3949** | **0.2742** | 0.6301 | +2.20% |
| AutoCF (ssl=0.05) | 0.3812 | 0.2683 | 0.6056 | 0.00% |
| AutoCF (ssl=0.1) | 0.3735 | 0.2633 | 0.5930 | -1.86% |

### Table 2: Performance on ML-1M (Dense Dataset)

| Model | HR@5 | NDCG@5 | HR@20 | Improv. vs LightGCN (NDCG@5) |
| :--- | :---: | :---: | :---: | :---: |
| **NeuMF** | **0.2981** | **0.1915** | **0.6759** | +20.5% (vs LightGCN) |
| LightGCN | 0.2481 | 0.1589 | 0.5688 | - |
| AutoCF (ssl=0.05) | 0.2372 | 0.1521 | 0.5187 | -4.28% |
| AutoCF (ssl=0.1) | 0.2632 | 0.1695 | 0.5931 | +6.67% |
| **AutoCF (ssl=0.125)** | **0.2643** | **0.1697** | **0.5953** | +6.80% |
| AutoCF (ssl=0.15) | 0.2608 | 0.1672 | 0.5804 | +5.22% |
###需要下载ML-1M和Grocery_and_Gourmet_Food数据集
###利用训练过的模型（上命令行例中为ML-1M_LightGCN_emb64.pt进行训练）
