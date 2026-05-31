import torch
from torch import nn
import numpy as np

def compute_selection_probabilities(A):
    """
    计算矩阵A中每一行和每一列被选中的概率（基于能量/L2范数平方）。
    """
    column_norms_squared = torch.sum(A**2, axis=0)
    row_norms_squared = torch.sum(A**2, axis=1)
    total_sum_squares = torch.sum(column_norms_squared)

    column_probs = column_norms_squared / (total_sum_squares + 1e-10)
    row_probs = row_norms_squared / (total_sum_squares + 1e-10)

    return column_probs, row_probs

def select_indices(probs, k, strategy='normal', replace=True):
    """
    根据不同策略选择索引。
    strategy: 'normal' (能量比例), 'inverse' (能量逆比例/探索), 'random' (均匀随机)
    """
    n = len(probs)
    if strategy == 'random':
        return np.random.choice(n, size=k, replace=replace)
    
    if strategy == 'inverse':
        # 逆概率策略：能量越小的越容易被选中
        inverted_P = (1 / (probs + 0.001)).float()
        p = inverted_P / inverted_P.sum()
    else: # 'normal'
        p = probs
        
    # numpy 不能直接处理 BFloat16 类型的 CUDA tensor，这里统一先转为 float32
    # 使用 float() 转换为 float32，再 detach().cpu().numpy()
    p = p.detach().cpu().float().numpy()
    p = p / p.sum() # 确保和为1
    
    return np.random.choice(n, size=k, replace=replace, p=p)

def adjust_duplicates(selected_indices, A, axis):
    """
    处理有放回抽样产生的重复项，进行缩放以保证无偏性。
    """
    unique_indices, counts = np.unique(selected_indices, return_counts=True)
    adjusted_matrix = A[:, unique_indices] if axis == 1 else A[unique_indices, :]
    
    for idx, count in enumerate(counts):
        if count > 1:
            scaling_factor = np.sqrt(count)
            if axis == 1:
                adjusted_matrix[:, idx] *= scaling_factor
            else:
                adjusted_matrix[idx, :] *= scaling_factor
    return adjusted_matrix, unique_indices

class CURModule(nn.Module):
    def __init__(self, W, rank=None, rank_c=None, rank_r=None,
                 train_C=False, train_U=True, train_R=False,
                 sampling_strategy='normal', replace=True, adjust_dups=True,
                 u_init='zero', dropout=0.0):
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
        """
        super(CURModule, self).__init__()
        
        # 确定 rank_c 和 rank_r
        if rank is not None:
            rank_c = rank_r = rank
        elif rank_c is None or rank_r is None:
            raise ValueError("Must provide either 'rank' or both 'rank_c' and 'rank_r'")

        # 1. 计算采样概率
        col_probs, row_probs = compute_selection_probabilities(W)
        
        # 2. 选择索引
        selected_cols = select_indices(col_probs, rank_c, strategy=sampling_strategy, replace=replace)
        selected_rows = select_indices(row_probs, rank_r, strategy=sampling_strategy, replace=replace)
        
        # 3. 提取 C 和 R 矩阵并处理重复项
        if replace and adjust_dups:
            C_init, unique_cols = adjust_duplicates(selected_cols, W, axis=1)
            R_init, unique_rows = adjust_duplicates(selected_rows, W, axis=0)
        else:
            C_init = W[:, selected_cols].detach().clone()
            R_init = W[selected_rows, :].detach().clone()
            unique_cols = selected_cols
            unique_rows = selected_rows

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
        self.selected_cols = selected_cols
        self.selected_rows = selected_rows

    def forward(self, x):
        # W_approx = C @ U @ R
        W_approx = torch.matmul(torch.matmul(self.C, self.U), self.R)
        # 线性层操作
        out = torch.matmul(x, W_approx.t())
        return self.dropout(out)

class LinearWithCURLoRA(nn.Module):
    """
    通用包装类，将任何线性层替换为 原始线性层 + CUR-LoRA 分支。
    """
    def __init__(self, linear, rank, alpha, **kwargs):
        super(LinearWithCURLoRA, self).__init__()
        self.linear = linear
        self.curlora = CURModule(linear.weight, rank, **kwargs)
        self.alpha = alpha

    def forward(self, x):
        # 原始输出 + alpha * CUR近似输出
        return self.linear(x) + self.alpha * self.curlora(x)
