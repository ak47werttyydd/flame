import os
import re

# 正则：去 ANSI 颜色
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# 正则：提取 step / loss
STEP_LOSS_RE = re.compile(r'step:\s*(\d+)\s+loss:\s*([0-9.]+)')

def parse_stderr_log(log_path):
    """解析单个 stderr.log，返回 {step: loss}"""
    step_loss = {}

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            clean = ANSI_ESCAPE.sub('', line)
            m = STEP_LOSS_RE.search(clean)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                step_loss[step] = loss

    return step_loss


def parse_all_ranks(root_dir):
    """
    遍历 root_dir 下所有数字命名的 rank 文件夹
    返回：rank -> {step -> loss}
    """
    rank_map = {}

    for name in os.listdir(root_dir):
        if not name.isdigit():
            continue

        rank = int(name)
        log_path = os.path.join(root_dir, name, "stderr.log")

        if not os.path.isfile(log_path):
            continue

        rank_map[rank] = parse_stderr_log(log_path)

    return rank_map


def print_rank_steps_loss(num_step,rank_step_loss):
    for step in range(1,num_step+1,1):
        for rank,step_loss in rank_step_loss.items():
            print(f"rank {rank}, step {step}, loss {step_loss[step]}")


rank_step_loss=parse_all_ranks("exp/gdn-340M-4K-20B/batch1.seqlen4096.context4096.warmup1024.update1.steps135633.lr3e-4.cosine/logs/none_nnw03ofw/attempt_0/")
avg_step_loss=parse_stderr_log("exp/gdn-340M-4K-20B/batch1.seqlen4096.context4096.warmup1024.update1.steps135633.lr3e-4.cosine.log")
# print(f"size of rank is {len(rank_step_loss)}, size of step are {len(rank_step_loss[0])},{len(rank_step_loss[1])},{len(rank_step_loss[2])},{len(rank_step_loss[3])}")
# print(f"On average, size of step is {len(avg_step_loss)}")
print_rank_steps_loss(100,rank_step_loss)