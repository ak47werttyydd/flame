import os
import re
from typing import Dict, Tuple

# 正则：去 ANSI 颜色
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

# 正则：提取 step / loss
STEP_LOSS_RE = re.compile(r'step:\s*(\d+)\s+loss:\s*([0-9.]+)')

# 正则：提取 step / local_gnorm (from gnorm_rank*.txt)
STEP_LOCAL_GNORM_RE = re.compile(r'step=(\d+),\s*local_gnorm_sq=([0-9.]+)')

# 正则：提取 step / global_gnorm (from gnorm_global.txt)
STEP_GLOBAL_GNORM_RE = re.compile(r'step=(\d+),\s*global_gnorm=([0-9.]+)') # should be global_gnorm_sq, but some legacy log uses global_gnorm


def parse_stderr_log(log_path: str) -> Dict[int, float]:
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


def parse_all_ranks(root_dir: str) -> Dict[int, Dict[int, float]]:
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


def parse_gnorm_rank_file(gnorm_path: str) -> Dict[int, float]:
    """
    解析单个 gnorm_rank*.txt 文件
    返回 {step: local_gnorm_sq}
    
    文件格式: step=123, local_gnorm=0.123456
    """
    step_gnorm = {}
    
    with open(gnorm_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = STEP_LOCAL_GNORM_RE.search(line)
            if m:
                step = int(m.group(1))
                gnorm_sq = float(m.group(2))
                step_gnorm[step] = gnorm_sq
    return step_gnorm


def parse_gnorm_global_file(gnorm_path: str) -> Dict[int, float]:
    """
    解析 gnorm_global.txt 文件
    返回 {step: global_gnorm_sq}
    
    文件格式: step=123, global_gnorm=0.123456
    """
    step_gnorm = {}
    
    with open(gnorm_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = STEP_GLOBAL_GNORM_RE.search(line)
            if m:
                step = int(m.group(1))
                gnorm_sq = float(m.group(2))
                step_gnorm[step] = gnorm_sq
    
    return step_gnorm


def parse_all_rank_gnorms(project_dir: str) -> Dict[int, Dict[int, float]]:
    """
    遍历 project_dir 下所有 gnorm_rank*.txt 文件
    返回：rank -> {step -> local_gnorm_sq}
    """
    rank_gnorm_map = {}
    
    # 正则匹配 gnorm_rank0.txt, gnorm_rank1.txt, ...
    gnorm_file_re = re.compile(r'gnorm_rank(\d+)\.txt')
    
    for name in os.listdir(project_dir):
        m = gnorm_file_re.match(name)
        if not m:
            continue
        
        rank = int(m.group(1))
        gnorm_path = os.path.join(project_dir, name)
        
        if not os.path.isfile(gnorm_path):
            continue
        
        rank_gnorm_map[rank] = parse_gnorm_rank_file(gnorm_path)
    
    return rank_gnorm_map


def parse_global_gnorm(project_dir: str) -> Dict[int, float]:
    """
    解析 project_dir 下的 gnorm_global.txt 文件
    返回：{step -> global_gnorm_sq}
    """
    gnorm_path = os.path.join(project_dir, "gnorm_global.txt")
    
    if not os.path.isfile(gnorm_path):
        return {}
    
    return parse_gnorm_global_file(gnorm_path)


def compute_critical_batch_size(
    rank_step_gnorm: Dict[int, Dict[int, float]],
    global_step_gnorm: Dict[int, float],
    B_small: int,
    num_cards: int,
    alpha: float = 0.1,
    beta: float = 0.1,
    num_steps: int = None
) -> Dict[int, Tuple[float, float, float, float]]:
    """
    计算逐步的 Critical Batch Size
    
    参数:
        rank_step_gnorm: rank -> {step -> local_gnorm_sq}，即 |G_{B_small,t}|^2
        global_step_gnorm: {step -> global_gnorm_sq}，即 |G_{B_big,t}|^2
        B_small: 单卡 batch size
        num_cards: 卡数
        alpha: EWMA 衰减参数 for |G|^2
        beta: EWMA 衰减参数 for S
        num_steps: 计算的步数，默认为 None 表示计算所有可用步数
    
    返回:
        {step -> (B_crit, G_sq_ewma, S_ewma, G_sq_instant, S_instant)}
    
    公式:
        B_big = B_small * num_cards
        
        |G_t|^2 = (B_big * |G_{B_big,t}|^2 - B_small * |G_{B_small,t}|^2) / (B_big - B_small)
        S_t = (|G_{B_small,t}|^2 - |G_{B_big,t}|^2) / (1/B_small - 1/B_big)
        
        EWMA:
        |G_t|^2_ewma = (1-alpha) * |G_{t-1}|^2_ewma + alpha * |G_t|^2
        S_t_ewma = (1-beta) * S_{t-1}_ewma + beta * S_t
        
        B_crit,t = S_t_ewma / |G_t|^2_ewma
    """
    B_big = B_small * num_cards
    
    # 计算 |G_{B_small,t}|^2: 取所有 rank 的平均值
    # 注意：在分布式训练中，每个 rank 的 local gradient 可能略有不同
    # 这里我们取平均作为 |G_{B_small}|^2 的估计
    def get_avg_local_gnorm_sq(step: int) -> float:
        gnorms = []
        for rank, step_gnorm in rank_step_gnorm.items():
            if step in step_gnorm:
                gnorms.append(step_gnorm[step])
        if not gnorms:
            return None
        return sum(gnorms) / len(gnorms)
    
    # 找出所有有效的 step（同时有 local 和 global gnorm 的步）
    all_steps = set(global_step_gnorm.keys())
    for rank, step_gnorm in rank_step_gnorm.items():
        all_steps &= set(step_gnorm.keys())

    if not all_steps:
        print("Warning: No common steps found between rank gnorms and global gnorm")
        return {}
    
    sorted_steps = sorted(all_steps)
    if num_steps is not None:
        sorted_steps = sorted_steps[:num_steps]
    
    results = {}
    G_sq_ewma = None
    S_ewma = None
    
    for step in sorted_steps:
        G_small_sq = get_avg_local_gnorm_sq(step)  # |G_{B_small,t}|^2
        G_big_sq = global_step_gnorm[step]          # |G_{B_big,t}|^2
        
        if G_small_sq is None:
            continue
        
        # 计算瞬时值 |G_t|^2 和 S_t
        # |G_t|^2 = (B_big * G_big_sq - B_small * G_small_sq) / (B_big - B_small)
        G_sq_instant = (B_big * G_big_sq - B_small * G_small_sq) / (B_big - B_small)
        
        # S_t = (G_small_sq - G_big_sq) / (1/B_small - 1/B_big)
        #     = (G_small_sq - G_big_sq) * B_small * B_big / (B_big - B_small)
        S_instant = (G_small_sq - G_big_sq) * B_small * B_big / (B_big - B_small)
        
        # EWMA 更新
        if G_sq_ewma is None:
            # 第一步：直接赋值
            G_sq_ewma = G_sq_instant
            S_ewma = S_instant
        else:
            # 后续步：EWMA 更新
            G_sq_ewma = (1 - alpha) * G_sq_ewma + alpha * G_sq_instant
            S_ewma = (1 - beta) * S_ewma + beta * S_instant
        
        # 计算 B_crit
        if G_sq_ewma > 0:
            B_crit = S_ewma / G_sq_ewma
        else:
            B_crit = float('inf')
        
        results[step] = (B_crit, G_sq_ewma, S_ewma, G_sq_instant, S_instant)
    
    return results


def print_rank_steps_loss(num_step: int, rank_step_loss: Dict[int, Dict[int, float]]):
    for step in range(1, num_step + 1, 1):
        for rank, step_loss in rank_step_loss.items():
            if step in step_loss:
                print(f"rank {rank}, step {step}, loss {step_loss[step]}")


def print_rank_steps_gnorm(num_step: int, rank_step_gnorm: Dict[int, Dict[int, float]]):
    """打印每个 rank 每个 step 的 local gnorm"""
    for step in range(1, num_step + 1, 1):
        for rank, step_gnorm in sorted(rank_step_gnorm.items()):
            if step in step_gnorm:
                print(f"rank {rank}, step {step}, local_gnorm_sq {step_gnorm[step]:.6f}")


def print_global_steps_gnorm(num_step: int, global_step_gnorm: Dict[int, float]):
    """打印每个 step 的 global gnorm"""
    for step in range(1, num_step + 1, 1):
        if step in global_step_gnorm:
            print(f"step {step}, global_gnorm_sq {global_step_gnorm[step]:.6f}")


def print_critical_batch_size(results: Dict[int, Tuple[float, float, float, float, float]], 
                               num_step: int = None, interval: int = 1):
    """
    打印 Critical Batch Size 结果
    
    results: {step -> (B_crit, G_sq_ewma, S_ewma, G_sq_instant, S_instant)}
    """
    sorted_steps = sorted(results.keys())
    if num_step is not None:
        sorted_steps = [s for s in sorted_steps if s <= num_step and (s % interval == 0)]
    
    print("=" * 100)
    print(f"{'Step':>6} | {'B_crit':>12} | {'|G|^2_ewma':>14} | {'S_ewma':>14} | {'|G|^2_inst':>14} | {'S_inst':>14}")
    print("-" * 100)
    
    for step in sorted_steps:
        B_crit, G_sq_ewma, S_ewma, G_sq_instant, S_instant = results[step]
        print(f"{step:>6} | {B_crit:>12.2f} | {G_sq_ewma:>14.6e} | {S_ewma:>14.6e} | {G_sq_instant:>14.6e} | {S_instant:>14.6e}")


# ==================== Main ====================

if __name__ == "__main__":
    # 配置参数
    project_dir = "exp/gdn-340M-4K-20B/bsize10.seqlen4096.context4096.warmup1024.update1.steps122070.DDPNoWrapper.Bcrit.lr3e-4.cosine"
    # logs_dir = f"{project_dir}/logs/none_nnw03ofw/attempt_0/"
    # avg_log_path = f"{project_dir}.log"
    
    # 批量大小配置
    B_small = 10  # 单卡 batch size（根据实际情况修改）
    num_cards = 4  # 卡数（根据实际情况修改）
    
    # EWMA 参数
    alpha = 0.02  # |G|^2 的衰减参数
    beta = 0.005  # S 的衰减参数
    
    # 解析 loss
    # rank_step_loss = parse_all_ranks(logs_dir)
    # avg_step_loss = parse_stderr_log(avg_log_path)
    
    # 解析 gnorm
    rank_step_gnorm = parse_all_rank_gnorms(project_dir)
    global_step_gnorm = parse_global_gnorm(project_dir)
    
    # print(f"Parsed {len(rank_step_loss)} ranks for loss")
    for rank, step_gnorm in sorted(rank_step_gnorm.items()):
        print(f"Rank {rank} has {len(step_gnorm)} steps of local gnorm")
    print(f"Parsed {len(global_step_gnorm)} steps for global gnorm")
    
    # 打印部分 loss
    # print("\n--- Loss (first 10 steps) ---")
    # print_rank_steps_loss(10, rank_step_loss)
    
    # 打印部分 gnorm
    # print("\n--- Local Gnorm (first 10 steps) ---")
    # print_rank_steps_gnorm(10, rank_step_gnorm)
    
    # print("\n--- Global Gnorm (first 10 steps) ---")
    # print_global_steps_gnorm(10, global_step_gnorm)
    
    # 计算 Critical Batch Size
    if rank_step_gnorm and global_step_gnorm:
        print("\n--- Critical Batch Size ---")
        print(f"B_small = {B_small}, B_big = {B_small * num_cards}, num_cards = {num_cards}")
        print(f"EWMA alpha = {alpha}, beta = {beta}")
        
        cbs_results = compute_critical_batch_size(
            rank_step_gnorm=rank_step_gnorm,
            global_step_gnorm=global_step_gnorm,
            B_small=B_small,
            num_cards=num_cards,
            alpha=alpha,
            beta=beta,
            num_steps=122070  # 计算前 100 步
        )
        
        print_critical_batch_size(cbs_results, num_step=122070, interval=1000)
    else:
        print("\nWarning: Cannot compute Critical Batch Size - missing gnorm data")
        print(f"  rank_step_gnorm has {len(rank_step_gnorm)} ranks")
        print(f"  global_step_gnorm has {len(global_step_gnorm)} steps")