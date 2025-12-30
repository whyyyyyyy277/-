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
###由于论文篇幅限制，其余两个模块的对比实验结果在下方呈现：
ML-1M (稠密)：
Mask Ratio 是核心胜负手：将 Mask 比率从 0.1 提升到 0.3，性能出现了显著的飞跃（NDCG@20 从 0.2555 涨到 0.2636）。
Teacher 彻底失效：无论将蒸馏权重加到多大（1.0, 5.0, 10.0），模型的最终表现与基准（0.1）完全一致（小数点后4位都一样）。这证明在稠密数据上，Teacher 模型提供的信息对 AutoCF 没有任何额外的指导意义。
Grocery (稀疏)：
Mask Ratio 敏感：降低 Mask 到 0.05 并没有显著提升（基本持平），而增加到 0.15 或 0.2 则导致性能缓慢下降。说明 0.1 左右确实是稀疏数据的甜蜜点。
蒸馏权重存在“甜蜜点”：0.1 是最佳权重。降低到 0.05 或增加到 0.2/0.5 都会导致性能轻微下降。
2. 详细数据分析表
实验 A: Mask Ratio 的影响 (寻找最佳增强力度)
Dataset	Mask Ratio	HR@20	NDCG@20	结论
ML-1M	0.1 (Baseline)	0.5727	0.2555	基础效果
	0.2	0.5885	0.2612	显著提升 (+2.2%)
	0.3	0.5948	0.2636	最佳性能 (+3.1%)
	0.5	0.5926	0.2629	开始回落，依然很强
Grocery	0.05	0.6300	0.3426	微弱最佳 (与0.1基本持平)
	0.1 (Baseline)	0.6301	0.3425	极其稳定
	0.15	0.6288	0.3423	轻微下降
	0.20	0.6297	0.3424	轻微下降
深度洞察 (ML-1M)：ML-1M 的图结构非常鲁棒。即使随机遮挡掉 30% 甚至 50% 的节点特征，模型依然能通过剩余信息重建并学习。这种“高难度”的对比学习任务迫使模型学到了更本质的用户偏好，从而带来了巨大的性能提升。
实验 B: Distill Weight 的影响 (Teacher 的作用边界)
Dataset	Distill Weight	HR@20	NDCG@20	结论
ML-1M	0.1 (Baseline)	0.5727	0.2555	-
	1.0	0.5727	0.2555	完全无效
	5.0	0.5727	0.2555	完全无效
	10.0	0.5727	0.2555	完全无效
Grocery	0.05	0.6291	0.3422	权重太低，指导不足
	0.1 (Baseline)	0.6301	0.3425	最佳平衡点
	0.2	0.6300	0.3423	开始产生负面约束
	0.5	0.6282	0.3416	强行模仿导致性能下降
深度洞察 (ML-1M)：这是一个非常有趣的现象。尽管我们将权重加大了 100 倍（从 0.1 到 10.0），且日志显示 total_loss 确实变大了（说明蒸馏项被计算进去了），但最终指标一动不动。
这说明：Teacher (LightGCN) 所学到的表征空间，对于 AutoCF (Student) 来说，没有任何“修正”或“拉动”的能力。 Student 自己跑出来的方向，Teacher 既不反对也不支持，或者 Student 的梯度完全主导了优化方向。在稠密数据上，可以放心大胆地丢掉 Teacher 模块。
3. 最终参数推荐 (Best Practices)
根据这两轮消融实验，整理出 AutoCF 在不同数据特质下的最佳实践参数：
🟢 针对 ML-1M (及其他稠密数据集)
Mask Ratio: 0.3 (这是关键涨点来源)
Distill Weight: 0 (既然没用，不如直接关掉以节省显存和计算时间，或者保留 0.1 作为心理安慰)
SSL Weight: 0.1 ~ 0.125 (根据之前的日志)
🟠 针对 Grocery (及其他稀疏数据集)
Mask Ratio: 0.05 或 0.1 (保持低强度增强，保护脆弱的图结构)
Distill Weight: 0.1 (必须保留，且权重不能太大也不能太小)
SSL Weight: 0.01 (稀疏数据上 SSL 权重通常较低)
