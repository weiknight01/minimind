from transformers import PretrainedConfig
from typing import List


class LMConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dim: int = 512, #模型维度，维度越大，模型越大，效果越好，但是速度越慢，内存占用越大。
            n_layers: int = 8,#层数，8或者12，层数越多，模型越大，效果越好，但是速度越慢，内存占用越大。
            n_heads: int = 8,#多头注意力机制中的“头数”，8或者16，确保dim能被n_heads整除。数量越大捕捉模型特征越强。
            n_kv_heads: int = 2,#
            vocab_size: int = 6400,#词表大小，决定模型能处理的不同词汇数量。较小的词表（如6400）适用于特定领域任务，通用模型通常使用更大的词表（如数万至百万级）
            hidden_dim: int = None,#不启用隐藏层
            multiple_of: int = 64,#默认值，约束中间层
            norm_eps: float = 1e-5,#默认值，较为平衡设置，没有必要不需要更改
            max_seq_len: int = 8192,#定义了模型能够处理的输入序列的最大长度（以 token 数量计），默认值为 8192。这一参数直接影响模型对长文本、代码、对话等场景的适配能力。
            rope_theta: int = 1e6,#长度大小决定处理文本长度能力，对话类
            dropout: float = 0.0,#默认值，dropout是一种正则化技术，用于防止过拟合。dropout=0.0表示不使用dropout。
            flash_attn: bool = True,#默认值，是否启用flash attention，flash attention是一种加速的注意力机制，可以提高模型的训练速度。
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,#模型较小，不使用MOE模式，降低训练资源消耗
            ####################################################
            num_experts_per_tok: int = 2,#每个token选择的专家数量
            n_routed_experts: int = 4,#总的专家数量
            n_shared_experts: bool = True,#共享专家
            scoring_func: str = 'softmax',#评分函数，默认为'softmax'
            aux_loss_alpha: float = 0.1,#辅助损失的alpha参数
            seq_aux: bool = True,  # 是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,#是否标准化top-k概率
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid#模型较小，不使用MOE模式，降低训练资源消耗
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率
        super().__init__(**kwargs)
