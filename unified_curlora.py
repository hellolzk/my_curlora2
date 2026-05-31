import torch
from torch import nn
import numpy as np


VALID_SAMPLING_STRATEGIES = {"normal", "inverse", "random"}
VALID_U_INITS = {"zero", "kaiming"}


def compute_selection_probabilities(A):
    """
    计算矩阵A中每一行和每一列被选中的概率（基于能量/L2范数平方）。
    """
    A_f32 = A.detach().float()
    column_norms_squared = torch.sum(A_f32**2, dim=0)
    row_norms_squared = torch.sum(A_f32**2, dim=1)
    total_sum_squares = torch.sum(column_norms_squared)

    column_probs = column_norms_squared / (total_sum_squares + 1e-10)
    row_probs = row_norms_squared / (total_sum_squares + 1e-10)

    return column_probs, row_probs


def _build_sampling_distribution(probs, strategy='normal'):
    """
    将能量概率转换为 numpy 采样分布，并在退化情况下回退到均匀分布。
    """
    if strategy not in VALID_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Invalid sampling strategy '{strategy}'. "
            f"Expected one of {sorted(VALID_SAMPLING_STRATEGIES)}."
        )

    n = len(probs)
    if n == 0:
        raise ValueError("Cannot sample from an empty probability vector.")

    if strategy == 'random':
        return np.full(n, 1.0 / n, dtype=np.float64)

    probs_np = probs.detach().cpu().float().numpy().astype(np.float64)
    if strategy == 'inverse':
        # 逆概率策略：能量越小的越容易被选中。
        probs_np = 1.0 / (probs_np + 1e-3)

    probs_np = np.nan_to_num(probs_np, nan=0.0, posinf=0.0, neginf=0.0)
    total_prob = probs_np.sum()
    if total_prob <= 0:
        return np.full(n, 1.0 / n, dtype=np.float64)

    return probs_np / total_prob


def select_indices(probs, k, strategy='normal', replace=True, rng=None):
    """
    根据不同策略选择索引。
    strategy: 'normal' (能量比例), 'inverse' (能量逆比例/探索), 'random' (均匀随机)
    """
    n = len(probs)
    if k is None or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}.")
    if not replace and k > n:
        raise ValueError(f"Cannot sample k={k} items without replacement from n={n} items.")

    sampling_probs = _build_sampling_distribution(probs, strategy=strategy)
    random_source = rng if rng is not None else np.random
    return random_source.choice(n, size=k, replace=replace, p=sampling_probs)


def extract_scaled_samples(selected_indices, A, axis, sampling_probs,
                           sample_count, adjust_dups=True, scale_by_prob=True):
    """
    提取被采样的列或行，并按 CUR 采样概率做缩放。

    当 adjust_dups=True 时，重复索引会被合并为一个索引，并乘以 sqrt(count)；
    当 scale_by_prob=True 时，每个样本再除以 sqrt(sample_count * p_i)。
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 (rows) or 1 (columns), got {axis}.")

    if adjust_dups:
        effective_indices, counts = np.unique(selected_indices, return_counts=True)
    else:
        effective_indices = np.asarray(selected_indices)
        counts = np.ones(len(effective_indices), dtype=np.int64)

    sampled = A[:, effective_indices] if axis == 1 else A[effective_indices, :]
    sampled = sampled.detach().clone()

    scale_values = np.sqrt(counts.astype(np.float64))
    if scale_by_prob:
        selected_probs = np.asarray(sampling_probs, dtype=np.float64)[effective_indices]
        selected_probs = np.maximum(selected_probs, 1e-12)
        scale_values = scale_values / np.sqrt(sample_count * selected_probs)

    scale_tensor = torch.as_tensor(scale_values, device=A.device, dtype=A.dtype)
    if axis == 1:
        sampled = sampled * scale_tensor.view(1, -1)
    else:
        sampled = sampled * scale_tensor.view(-1, 1)

    return sampled, effective_indices, counts


class CURModule(nn.Module):
    def __init__(self, W, rank=None, rank_c=None, rank_r=None,
                 train_C=False, train_U=True, train_R=False,
                 sampling_strategy='normal', replace=True, adjust_dups=True,
                 u_init='zero', dropout=0.0, seed=None, scale_by_prob=True):
        """
        集成的 CUR 模块。
        
        参数:
            W: 原始权重矩阵 (out_features, in_features)。
            rank: 统一的秩（如果设置，则 rank_c 和 rank_r 都等于此值）。
            rank_c: C 矩阵选取的列数。
            rank_r: R 矩阵选取的行数。
            train_C/U/R: 是否对 C, U, R 进行微调。
            sampling_strategy: 'normal', 'inverse', 'random'。
            replace: 是否有放回抽样。
            adjust_dups: 在有放回抽样时是否进行重复项缩放。
            u_init: U 的初始化方式 'zero' 或 'kaiming'。
            dropout: CUR 分支的 Dropout 概率。
            seed: numpy 采样随机种子；为 None 时使用全局 numpy 随机状态。
            scale_by_prob: 是否按 CUR 采样概率缩放 C 和 R。
        """
        super(CURModule, self).__init__()
        
        # 确定 rank_c 和 rank_r
        if rank is not None:
            rank_c = rank_r = rank
        elif rank_c is None or rank_r is None:
            raise ValueError("Must provide either 'rank' or both 'rank_c' and 'rank_r'")
        if rank_c <= 0 or rank_r <= 0:
            raise ValueError(f"rank_c and rank_r must be positive, got rank_c={rank_c}, rank_r={rank_r}.")
        if sampling_strategy not in VALID_SAMPLING_STRATEGIES:
            raise ValueError(
                f"Invalid sampling strategy '{sampling_strategy}'. "
                f"Expected one of {sorted(VALID_SAMPLING_STRATEGIES)}."
            )
        if u_init not in VALID_U_INITS:
            raise ValueError(f"Invalid u_init '{u_init}'. Expected one of {sorted(VALID_U_INITS)}.")

        # 1. 计算采样概率
        col_probs, row_probs = compute_selection_probabilities(W)
        col_sampling_probs = _build_sampling_distribution(col_probs, strategy=sampling_strategy)
        row_sampling_probs = _build_sampling_distribution(row_probs, strategy=sampling_strategy)
        rng = np.random.default_rng(seed) if seed is not None else None
        
        # 2. 选择索引
        selected_cols = select_indices(col_probs, rank_c, strategy=sampling_strategy, replace=replace, rng=rng)
        selected_rows = select_indices(row_probs, rank_r, strategy=sampling_strategy, replace=replace, rng=rng)
        
        # 3. 提取 C 和 R 矩阵并处理重复项
        C_init, unique_cols, col_counts = extract_scaled_samples(
            selected_cols,
            W,
            axis=1,
            sampling_probs=col_sampling_probs,
            sample_count=rank_c,
            adjust_dups=replace and adjust_dups,
            scale_by_prob=scale_by_prob,
        )
        R_init, unique_rows, row_counts = extract_scaled_samples(
            selected_rows,
            W,
            axis=0,
            sampling_probs=row_sampling_probs,
            sample_count=rank_r,
            adjust_dups=replace and adjust_dups,
            scale_by_prob=scale_by_prob,
        )

        # 4. 初始化 U 矩阵
        if train_U:
            # U 初始化为 0 或 Kaiming 高斯
            U_init = torch.zeros(C_init.shape[1], R_init.shape[0], device=W.device, dtype=W.dtype)
            if u_init == 'kaiming':
                nn.init.kaiming_uniform_(U_init, a=np.sqrt(5))
        else:
            # U 不被训练时，通过伪逆计算
            W_f32 = W.float()
            C_f32 = C_init.float()
            R_f32 = R_init.float()
            C_pinv = torch.pinverse(C_f32)
            R_pinv = torch.pinverse(R_f32)
            U_init = torch.matmul(torch.matmul(C_pinv, W_f32), R_pinv)
            U_init = U_init.to(W.dtype)

        # 5. 注册
        if train_C:
            self.C = nn.Parameter(C_init.detach().clone())
        else:
            self.register_buffer('C', C_init.detach().clone())
            
        if train_U:
            self.U = nn.Parameter(U_init.detach().clone())
        else:
            self.register_buffer('U', U_init.detach().clone())
            
        if train_R:
            self.R = nn.Parameter(R_init.detach().clone())
        else:
            self.register_buffer('R', R_init.detach().clone())

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 保存元数据
        self.rank_c = rank_c
        self.rank_r = rank_r
        self.sampling_strategy = sampling_strategy
        self.replace = replace
        self.adjust_dups = adjust_dups
        self.scale_by_prob = scale_by_prob
        self.actual_rank_c = C_init.shape[1]
        self.actual_rank_r = R_init.shape[0]
        self.selected_cols = selected_cols
        self.selected_rows = selected_rows
        self.unique_cols = unique_cols
        self.unique_rows = unique_rows
        self.col_counts = col_counts
        self.row_counts = row_counts

    def forward(self, x):
        # 等价于 x @ (C @ U @ R).T，但避免显式构造完整的权重矩阵。
        out = torch.matmul(x, self.R.t())
        out = torch.matmul(out, self.U.t())
        out = torch.matmul(out, self.C.t())
        return self.dropout(out)


class LinearWithCURLoRA(nn.Module):
    """
    通用包装类，将任何线性层替换为 原始线性层 + CUR-LoRA 分支。
    """
    def __init__(self, linear, rank=None, alpha=1.0, freeze_base=True, **kwargs):
        super(LinearWithCURLoRA, self).__init__()
        self.linear = linear
        if freeze_base:
            for param in self.linear.parameters():
                param.requires_grad = False
        self.curlora = CURModule(linear.weight, rank, **kwargs)
        self.alpha = alpha
        self.freeze_base = freeze_base

    def forward(self, x):
        # 原始输出 + alpha * CUR近似输出
        return self.linear(x) + self.alpha * self.curlora(x)
