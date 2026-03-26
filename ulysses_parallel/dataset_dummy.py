import torch, torch_npu

class LLMDummyDataset(torch.utils.data.Dataset):
    """训练 LLM 用的 dummy 数据集"""
    def __init__(self, vocab_size: int, seq_len: int = 32, size: int = 100):
        self.size = size
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # LLM 训练通常使用 input_ids 右移一位作为 labels
        # 此处简化直接生成 dummy labels
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return input_ids, labels