import torch, torch_npu

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, hidden_size, size=1000):
        # Here we only care about the Attention layer, so we skipped 
        # embedding layer and give hidden_size dim directly
        self.x = torch.randn(size, 4, hidden_size)
        self.y = torch.randint(0, 2, (size, hidden_size))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]