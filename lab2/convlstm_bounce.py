"""弹跳小球时空预测实验:ConvLSTM
输入前 4 帧, 预测第 5 帧。
"""
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

# ---------- 全局配置 ----------
SEED = 42
IMG = 32                 # 图像尺寸 32x32
RADIUS = 2               # 小球半径
N_FRAMES = 10            # 每条序列帧数
N_SEQ = 2000             # 训练序列数
N_TEST = 200             # 测试序列数
N_INPUT = 4              # 输入帧数(预测下一帧)
VEL_MIN, VEL_MAX = 0.8, 1.6   # 初始速度范围(像素/帧)

EPOCHS = 5
BATCH = 64
LR = 1e-3

# 模型可配置参数
HID_CH = 32              # 隐藏通道数
NUM_LAYERS = 2           # 层数
KERNEL = 5               # 卷积核大小

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 中文字体(Microsoft YaHei)
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ---------- 1. 数据合成 ----------
def make_bounce_data(n_seq, seed=SEED):
    """生成 n_seq 条弹跳小球序列, 返回 (n_seq, N_FRAMES, IMG, IMG) 的 float32 数组。"""
    rng = np.random.RandomState(seed)
    data = np.zeros((n_seq, N_FRAMES, IMG, IMG), dtype=np.float32)
    ys, xs = np.mgrid[0:IMG, 0:IMG]
    for s in range(n_seq):
        # 随机初始位置与速度
        x = rng.uniform(RADIUS, IMG - RADIUS)
        y = rng.uniform(RADIUS, IMG - RADIUS)
        speed = rng.uniform(VEL_MIN, VEL_MAX)
        angle = rng.uniform(0, 2 * np.pi)
        vx, vy = speed * np.cos(angle), speed * np.sin(angle)
        for t in range(N_FRAMES):
            # 更新位置并在边界反弹
            x += vx
            y += vy
            if x < RADIUS:
                x, vx = 2 * RADIUS - x, -vx
            elif x > IMG - RADIUS:
                x, vx = 2 * (IMG - RADIUS) - x, -vx
            if y < RADIUS:
                y, vy = 2 * RADIUS - y, -vy
            elif y > IMG - RADIUS:
                y, vy = 2 * (IMG - RADIUS) - y, -vy
            # 渲染小球(圆形掩膜)
            mask = (xs - x) ** 2 + (ys - y) ** 2 <= RADIUS ** 2
            data[s, t] = mask.astype(np.float32)
    return data


# ---------- 2. ConvLSTMCell ----------
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=KERNEL):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, k, padding=k // 2)

    def forward(self, x, h, c):
        # x:(B,in,H,W) h:(B,hid,H,W) -> cat 后过一个卷积, 输出 4*hid 通道
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


# ---------- 3. ConvLSTM 模型 ----------
class ConvLSTM(nn.Module):
    def __init__(self, in_ch=1, hid_ch=HID_CH, num_layers=NUM_LAYERS, k=KERNEL):
        super().__init__()
        self.num_layers = num_layers
        self.hid_ch = hid_ch
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            self.cells.append(ConvLSTMCell(in_ch if l == 0 else hid_ch, hid_ch, k))
        # 从最后隐藏状态预测下一帧
        self.out_conv = nn.Conv2d(hid_ch, 1, 3, padding=1)

    def forward(self, x):
        # x:(B, T, C, H, W)
        B, T, C, H, W = x.shape
        h = [torch.zeros(B, self.hid_ch, H, W, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hid_ch, H, W, device=x.device) for _ in range(self.num_layers)]
        for t in range(T):
            xt = x[:, t]
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l](xt, h[l], c[l])
                xt = h[l]
        return self.out_conv(h[-1])  # (B, 1, H, W)


def to_xy(data):
    """data:(N,T,H,W) -> x:(N,4,1,H,W), y:(N,1,H,W)"""
    x = torch.from_numpy(data[:, :N_INPUT, None])
    y = torch.from_numpy(data[:, N_INPUT, None])
    return x, y


def main():
    torch.manual_seed(SEED)
    t0 = time.time()
    print(f"设备: {DEVICE}")

    # ---------- 数据: 一次生成 2000+200 条(种子 42), 前 2000 训练, 后 200 测试 ----------
    all_data = make_bounce_data(N_SEQ + N_TEST, SEED)
    x_train, y_train = to_xy(all_data[:N_SEQ])
    x_test, y_test = to_xy(all_data[N_SEQ:])
    print(f"训练样本: {x_train.shape[0]}, 测试样本: {x_test.shape[0]}")

    # ---------- 训练 ----------
    model = ConvLSTM().to(DEVICE)

    # ---------- 参数量统计(对照手算) ----------
    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    def hand_calc_cell(in_ch, hid_ch):
        # Conv2d(in+hid, 4*hid, k=KERNEL, padding=k//2): 权重 4*hid*(in+hid)*K^2 + 偏置 4*hid
        return 4 * hid_ch * (in_ch + hid_ch) * KERNEL * KERNEL + 4 * hid_ch

    n_cell = sum(count_params(cell) for cell in model.cells)
    n_out = count_params(model.out_conv)
    n_total = count_params(model)
    n_cell_hand = sum(hand_calc_cell(1 if l == 0 else HID_CH, HID_CH) for l in range(NUM_LAYERS))
    print(f"ConvLSTMCell 参数量: {n_cell} (手算: {n_cell_hand})")
    print(f"输出卷积参数量: {n_out} (手算: {1 * HID_CH * 3 * 3 + 1})")
    print(f"总参数量: {n_total} (手算: {n_cell_hand + 1 * HID_CH * 3 * 3 + 1})")

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
    print(f"\n最终结果 -> 测试 MSE: {test_mses[-1]:.5f} | 测试 MAE: {test_maes[-1]:.5f} | 总耗时: {elapsed:.2f} 秒")

    # ---------- 4. 损失曲线 ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = range(1, EPOCHS + 1)
    ax.plot(epochs, train_losses, marker='o', label='训练 MSE')
    ax.plot(epochs, test_mses, marker='s', label='测试 MSE')
    ax.plot(epochs, test_maes, marker='^', label='测试 MAE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('ConvLSTM 弹跳小球预测训练曲线')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('result_baseline.png', dpi=150)
    plt.close(fig)
    print("已保存 result_baseline.png")

    # ---------- 5. 九宫格对比图: 3 样本 x (输入第4帧 | 真实帧 | 预测帧) ----------
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(DEVICE)).cpu()

    samples = [0, 50, 100]
    titles = [f'输入第 {N_INPUT} 帧', f'真实帧(第 {N_INPUT + 1} 帧)', '预测帧']
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for r, s in enumerate(samples):
        frames = [
            x_test[s, -1, 0].numpy(),   # 最后一个输入帧
            y_test[s, 0].numpy(),       # 真实下一帧
            pred[s, 0].numpy(),         # 预测帧
        ]
        for c, arr in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(arr, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c])
        axes[r, 0].set_ylabel(f'样本 {s}')
    fig.suptitle('输入 4 帧 → ConvLSTM 预测下一帧', y=1.02)
    fig.tight_layout()
    fig.savefig('result_pred.png', dpi=150)
    plt.close(fig)
    print("已保存 result_pred.png")


if __name__ == '__main__':
    main()
