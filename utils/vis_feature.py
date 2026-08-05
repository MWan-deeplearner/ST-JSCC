import matplotlib.pyplot as plt
import torch
import numpy as np


def visualize_feature_maps(
    x1: torch.Tensor,
    x2: torch.Tensor,
    save_path: str = "feature_heatmap.png",
    num_channels: int = 8,
    figsize: tuple = (16, 6),
):
    """可视化 GLBlock 的 x1 (LinearAttention) 和 x2 (Residual) 特征热图。

    对特征图在通道维度上取均值得到 2D 热图，并排显示。

    Args:
        x1: [B, C, H, W] — LinearAttentionModular 输出
        x2: [B, C, H, W] — ResidualModular 输出
        save_path: 保存图片路径
        num_channels: 单独显示的通道数（前 k 个通道）
        figsize: 整张画布大小
    """
    if x1.dim() == 4 and x1.size(0) > 1:
        # 多 batch 取第一个
        x1 = x1[0:1]
        x2 = x2[0:1]

    # 转 numpy
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()

    if x1_np.ndim == 4:
        x1_np = x1_np[0]  # [C, H, W]
        x2_np = x2_np[0]

    # ========== 1) 通道均值热图（全局响应） ==========
    x1_mean = x1_np.mean(axis=0)  # [H, W]
    x2_mean = x2_np.mean(axis=0)

    vmin = min(x1_mean.min(), x2_mean.min())
    vmax = max(x1_mean.max(), x2_mean.max())

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im0 = axes[0].imshow(x1_mean, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("x1 — LinearAttention (channel mean)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(x2_mean, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("x2 — ResidualModular (channel mean)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.suptitle("Feature Heatmaps (mean over channels)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_mean.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] Saved channel-mean heatmap -> {save_path.replace('.png', '_mean.png')}")

    # ========== 2) 前 k 个通道单独显示 ==========
    k = min(num_channels, x1_np.shape[0])
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6))
    if k == 1:
        axes = axes.reshape(2, 1)

    per_vmin = min(x1_np[:k].min(), x2_np[:k].min())
    per_vmax = max(x1_np[:k].max(), x2_np[:k].max())

    for i in range(k):
        axes[0, i].imshow(x1_np[i], cmap="viridis", vmin=per_vmin, vmax=per_vmax)
        axes[0, i].set_title(f"x1 ch-{i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(x2_np[i], cmap="viridis", vmin=per_vmin, vmax=per_vmax)
        axes[1, i].set_title(f"x2 ch-{i}")
        axes[1, i].axis("off")

    plt.suptitle(f"Feature Heatmaps (first {k} channels)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_per_channel.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] Saved per-channel heatmap -> {save_path.replace('.png', '_per_channel.png')}")


if __name__ == "__main__":
    # 简单测试
    dummy_x1 = torch.randn(1, 48, 32, 32)
    dummy_x2 = torch.randn(1, 48, 32, 32)
    visualize_feature_maps(dummy_x1, dummy_x2, "test_heatmap.png")