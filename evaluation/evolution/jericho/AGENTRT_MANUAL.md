# Agent-R EvolEval — 迁移运行手册

## 概览

Agent-R 是参数式 baseline：通过 QLoRA 微调将失败经验烧进权重，评测时 prompt 里不含任何历史 snapshot。完整 pipeline 分四阶段：

```
Build data → Fine-tune (QLoRA) → Merge LoRA → EvolEval
```

预计总耗时：**12–18 小时**（Qwen3-32B，4 × A100/H100）

---

## 1. 环境准备

### 1.1 克隆代码

```bash
git clone https://github.com/x686A68/ROGUE.git
cd ROGUE
```

### 1.2 创建 Conda 环境

```bash
conda create -n Fraud python=3.10 -y
conda activate Fraud
```

### 1.3 安装依赖

```bash
# Jericho 文字游戏引擎
pip install jericho

# LLM 推理 / 微调
pip install vllm
pip install transformers accelerate peft
pip install bitsandbytes trl

# 其他工具
pip install openai
```

> **注意**：`bitsandbytes` 需要 CUDA 11.8+，请确认驱动版本。

### 1.4 下载模型

```bash
# 方案 A：从 HuggingFace 下载（需要能访问 HF）
huggingface-cli download Qwen/Qwen3-32B --local-dir /path/to/models/Qwen3-32B

# 方案 B：直接用 HF model ID（vllm/transformers 会自动下载到缓存）
# 什么都不用做，脚本里的 "Qwen/Qwen3-32B" 会自动拉取
```

---

## 2. 数据准备

### 2.1 需要的文件

从原服务器把以下目录传到新服务器：

```
ROGUE/final_dataset/jericho/          ← 必须！dataset + evolution_evaluation
ROGUE/JTTL/EvoTest/jericho-games/    ← 必须！游戏 ROM 文件（.z3/.z5/.z8 等）
ROGUE/JTTL/EvolEval/                 ← 代码目录
```

传输命令（在原服务器上执行）：

```bash
rsync -avz \
  /home/jiahao_huang/ROGUE/final_dataset/jericho/ \
  user@NEW_SERVER:/path/to/ROGUE/final_dataset/jericho/

rsync -avz \
  /home/jiahao_huang/ROGUE/JTTL/EvoTest/jericho-games/ \
  user@NEW_SERVER:/path/to/ROGUE/JTTL/EvoTest/jericho-games/
```

### 2.2 验证数据完整性

```bash
cd /path/to/ROGUE/JTTL/EvolEval

conda run -n Fraud python -c "
from utils import DATASET_ROOT, ROM_DIR, GAMES, game_file
print('DATASET_ROOT:', DATASET_ROOT)
print('ROM_DIR:', ROM_DIR)
for g in GAMES:
    snap = DATASET_ROOT / g / 'snapshots.json'
    ee = DATASET_ROOT / g / 'evolution_evaluation.json'
    rom = ROM_DIR / game_file(g)
    ok = all(p.exists() for p in [snap, ee, rom])
    print(f'  {g}: {\"OK\" if ok else \"MISSING\"}')
"
```

所有游戏都应显示 `OK`。

---

## 3. 修改 Pipeline 脚本路径

`run_agentR_pipeline.sh` 里有几个硬编码路径，**必须改成新服务器的实际路径**：

```bash
# 编辑脚本
nano /path/to/ROGUE/JTTL/EvolEval/run_agentR_pipeline.sh
```

修改以下变量（脚本开头）：

```bash
# 改成新服务器上 conda 环境里的 python 路径
PYTHON=/path/to/miniconda3/envs/Fraud/bin/python

# 改成新服务器上 ROGUE 根目录
ROGUE=/path/to/ROGUE

# 如果模型放在本地路径，改这里
MODEL=/path/to/models/Qwen3-32B   # 或者保持 "Qwen/Qwen3-32B" 用 HF 缓存
```

查找 Python 路径：

```bash
conda run -n Fraud which python
```

---

## 4. 运行完整 Pipeline

**必须在 tmux 里启动，用 nohup 保底**：

```bash
# 开一个 tmux session
tmux new -s agentR

cd /path/to/ROGUE/JTTL/EvolEval
mkdir -p logs/agentR_pipeline

nohup bash run_agentR_pipeline.sh > logs/agentR_pipeline/main.log 2>&1 &
echo "PID: $!"
echo "tail -f logs/agentR_pipeline/main.log"
```

---

## 5. 分阶段手动运行（如果 pipeline 中途失败）

### Step 1 — 启动 vLLM（base model）

```bash
conda activate Fraud

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --served-model-name Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port 8000 \
    --trust-remote-code \
    --override-generation-config '{"enable_thinking": false}' \
    > logs/agentR_pipeline/vllm_base.log 2>&1 &

# 等待 ready（约 2-3 分钟）
until curl -sf http://localhost:8000/health; do sleep 5; done
echo "vLLM ready"
```

### Step 2 — 构建训练数据

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local

for game in balances detective library temple zork1 zork3; do
    python agentR_build_data.py \
        --game $game \
        --model Qwen/Qwen3-32B \
        --output agentR_data/${game}.jsonl \
        --n_workers 4 \
        > logs/agentR_pipeline/build_${game}.log 2>&1
    echo "$game done"
done
```

### Step 3 — 关掉 vLLM

```bash
pkill -f "vllm.entrypoints"
sleep 10
```

### Step 4 — QLoRA 微调

```bash
CUDA_VISIBLE_DEVICES=0 python agentR_finetune.py \
    --data agentR_data/balances.jsonl agentR_data/detective.jsonl \
            agentR_data/library.jsonl agentR_data/temple.jsonl \
            agentR_data/zork1.jsonl agentR_data/zork3.jsonl \
    --base_model Qwen/Qwen3-32B \
    --output_dir output/agentR_ckpt/all_games \
    --epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lr 2e-4 \
    --max_length 4096 \
    --batch_size 1 \
    --grad_accum 8 \
    > logs/agentR_pipeline/finetune.log 2>&1
```

### Step 5 — Merge LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python - \
    Qwen/Qwen3-32B \
    output/agentR_ckpt/all_games/adapter \
    output/agentR_merged \
    > logs/agentR_pipeline/merge.log 2>&1 << 'EOF'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id, adapter_path, merged_path = sys.argv[1], sys.argv[2], sys.argv[3]
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(merged_path)
AutoTokenizer.from_pretrained(base_id, trust_remote_code=True).save_pretrained(merged_path)
print("Merge complete.")
EOF
```

### Step 6 — 启动 vLLM（微调后模型）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model output/agentR_merged \
    --served-model-name agentR \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --port 8000 \
    --trust-remote-code \
    --override-generation-config '{"enable_thinking": false}' \
    > logs/agentR_pipeline/vllm_finetuned.log 2>&1 &

until curl -sf http://localhost:8000/health; do sleep 5; done
echo "Fine-tuned vLLM ready"
```

### Step 7 — EvolEval 评测

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=local

for game in balances detective library temple zork1 zork3; do
    python run_agentR.py \
        --game $game \
        --model agentR \
        --temperature 0.4 \
        > logs/agentR_pipeline/eval_${game}.log 2>&1
    echo "$game eval done"
done
```

### Step 8 — 关闭 vLLM（实验结束必须关）

```bash
pkill -f "vllm.entrypoints"
sleep 10
echo "vLLM stopped"
```

---

## 6. 监控进度

```bash
# 主日志
tail -f logs/agentR_pipeline/main.log

# 某个游戏的 build data 进度
tail -f logs/agentR_pipeline/build_zork1.log

# 训练 loss
grep "loss" logs/agentR_pipeline/finetune.log | tail -20

# 评测进度
tail -f logs/agentR_pipeline/eval_zork1.log

# GPU 使用率
watch -n 5 nvidia-smi
```

---

## 7. 结果位置

```
JTTL/EvolEval/results/<game>/agentR/<model>/<timestamp>/
├── results.json       ← 所有 task 的评测结果
└── <task_id>.log      ← 每个 task 的逐步轨迹
```

收集所有结果：

```bash
python score_evol_eval.py --baseline agentR
```

---

## 8. 常见问题

**Q: vLLM 启动失败，提示 CUDA OOM**
→ 检查 `--max-model-len`，改小到 `16384`；确认没有其他进程占用 GPU（`nvidia-smi`）

**Q: `bitsandbytes` 报错**
→ `pip install bitsandbytes --upgrade`；确认 CUDA version 与 PyTorch 匹配

**Q: `agentR_build_data.py` 某个游戏 0 examples**
→ 正常现象，说明该游戏所有 snapshot 没有可识别的错误步骤或纠错失败；只要总 examples > 0 就能训练

**Q: 微调后评测 score 比 Naive 更低**
→ 检查训练数据量（每个游戏至少需要 10+ examples），或尝试增加 epochs

**Q: 路径找不到 `final_dataset`**
→ 确认 `utils.py` 里 `DATASET_ROOT` 的相对路径是从 `JTTL/EvolEval/` 出发的，路径为 `../../final_dataset/jericho`
