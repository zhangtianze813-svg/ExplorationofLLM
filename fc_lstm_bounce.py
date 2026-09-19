"""结构对照: 全连接 LSTM(FC-LSTM) 弹跳小球预测
与 ConvLSTM 基线对照: 输入 4 帧(展平成 1024 维向量), 预测第 5 帧。
FC-LSTM 隐藏维取 32, 与 ConvLSTM 隐藏通道 32 对齐。
"""
import time
import torch
import torch.nn as nn

from convlstm_bounce import (IMG, N_FRAMES, N_SEQ, N_TEST, N_INPUT, SEED,
                             make_bounce_data)

EPOCHS = 5
BATCH = 64
LR = 1e-3
HID = 32                    # 与 ConvLSTM 隐藏通道数对齐

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_xy_flat(data):
    """data:(N,T,H,W) -> x:(N,4,1024), y:(N,1024) 展平后的帧向量"""
    x = torch.from_numpy(data[:, :N_INPUT].reshape(-1, N_INPUT, IMG * IMG))
    y = torch.from_numpy(data[:, N_INPUT].reshape(-1, IMG * IMG))
    return x, y


class FCLSTM(nn.Module):
    def __init__(self, in_dim=IMG * IMG, hid=HID):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, batch_first=True)
        self.out = nn.Linear(hid, in_dim)

    def forward(self, x):
        # x:(B, T, D)
        _, (h, _) = self.lstm(x)
        return self.out(h[-1])  # (B, D)


def main():
    torch.manual_seed(SEED)
    t0 = time.time()
    print(f"设备: {DEVICE}")

    # 与 ConvLSTM 完全相同的数据(种子 42, 前 2000 训练, 后 200 测试)
    all_data = make_bounce_data(N_SEQ + N_TEST, SEED)
    x_train, y_train = to_xy_flat(all_data[:N_SEQ])
    x_test, y_test = to_xy_flat(all_data[N_SEQ:])
    print(f"训练样本: {x_train.shape[0]}, 测试样本: {x_test.shape[0]}")

    model = FCLSTM().to(DEVICE)
    n_lstm = sum(p.numel() for p in model.lstm.parameters())
    n_out = sum(p.numel() for p in model.out.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    # 手算: LSTM = 4*hid*(in+hid+2) = 128*1058 = 135424; 输出层 = 32*1024 + 1024 = 33792
    print(f"LSTM 参数量: {n_lstm} (手算: {4 * HID * (IMG * IMG + HID + 2)})")
    print(f"输出层参数量: {n_out} (手算: {HID * IMG * IMG + IMG * IMG})")
    print(f"总参数量: {n_total} (手算: {4 * HID * (IMG * IMG + HID + 2) + HID * IMG * IMG + IMG * IMG})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_losses, test_mses, test_maes = [], [], []
    n = len(x_train)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total, cnt = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = x_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            cnt += len(idx)
        train_losses.append(total / cnt)

        model.eval()
        with torch.no_grad():
            pred = model(x_test.to(DEVICE))
            yb = y_test.to(DEVICE)
            mse = ((pred - yb) ** 2).mean().item()
            mae = (pred - yb).abs().mean().item()
        test_mses.append(mse)
        test_maes.append(mae)
        print(f"epoch {ep + 1:02d}/{EPOCHS} | train_mse {train_losses[-1]:.5f} "
              f"| test_mse {mse:.5f} | test_mae {mae:.5f}")

    elapsed = time.time() - t0
    print(f"\nFC-LSTM 结果 -> 测试 MSE: {test_mses[-1]:.5f} | 测试 MAE: {test_maes[-1]:.5f} | 总耗时: {elapsed:.2f} 秒")


if __name__ == '__main__':
    main()
