import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
methods_config = {
    'ResJSCC (2019)':       {'color': '#4682B4', 'marker': '^', 'linewidth': 1.5, 'zorder':  5},
    'FAJSCC (2026)':        {'color': '#D2691E', 'marker': 'v', 'linewidth': 1.5, 'zorder':  5},
    'ST-JSCC (w/o LAM)':    {'color': '#00CED1', 'marker': 'p', 'linewidth': 1.5, 'zorder':  5},
    'ST-JSCC (w/o RM)':     {'color': '#9370DB', 'marker': '*', 'linewidth': 1.5, 'zorder':  5},
    'ST-JSCC (w/o GFFN)':   {'color': '#9ACD32', 'marker': 'D', 'linewidth': 1.5, 'zorder':  5},
    'ST-JSCC (Full Model)': {'color': '#FF0000', 'marker': 's', 'linewidth': 2.5, 'zorder': 10},
}
markersize  =  7
label_size  = 16
tick_sizes  = 16
legend_size = 16
title_size  = 16
grid_alpha  = 0.7

def plot_2x3(dataset, all_data, snr_list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    rows = ["AWGN", "Rayleigh"]
    cols = ["PSNR", "SSIM", "MS-SSIM"]
    y_labels = ["PSNR (dB)$\\uparrow$", "SSIM$\\uparrow$", "MS-SSIM$\\uparrow$"]
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    idx = 0
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            ax = axes[j]
            key = f"{dataset}-{row}-{col}"
            if key not in all_data:
                continue
            data = all_data[key]
            for name, vals in data.items():
                cfg = methods_config.get(name)
                if cfg is None:
                    continue
                ax.plot(snr_list, vals, label=name if i==0 and j==0 else "",
                        color=cfg['color'], marker=cfg['marker'],
                        linewidth=cfg['linewidth'], markersize=markersize, zorder=cfg['zorder'])
            ax.set_xlabel("SNR (dB)",  fontsize=label_size)
            ax.set_ylabel(y_labels[j], fontsize=label_size)
            ax.set_xticks(snr_list)
            ax.tick_params(labelsize=tick_sizes)
            ax.grid(True, linestyle='--', alpha=grid_alpha)
            ax.set_title(f"{subplot_labels[idx]} {row} / {col}", fontsize=title_size)
            idx += 1
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=4, fontsize=legend_size, fancybox=True, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"ablation_{dataset}.pdf", bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    snr_points = [1, 4, 7, 10]

    dataset_list = ["DIV2K", "Kodak"]
    channel_list = ["AWGN",  "Rayleigh"]
    metrics_list = ["PSNR",  "SSIM", "MS-SSIM"]
    all_data = {
        "Kodak-AWGN-PSNR": {
            'ResJSCC (2019)':       [25.28, 26.28, 26.83, 27.13],
            'FAJSCC (2026)':        [25.91, 26.90, 27.49, 27.82],
            'ST-JSCC (w/o LAM)':    [26.03, 27.61, 28.61, 29.19],
            'ST-JSCC (w/o RM)':     [24.98, 26.57, 27.69, 28.39],
            'ST-JSCC (w/o GFFN)':   [25.33, 26.73, 27.59, 28.08],
            'ST-JSCC (Full Model)': [26.12, 27.75, 28.77, 29.36],
        },
        "Kodak-AWGN-SSIM": {
            'ResJSCC (2019)':       [0.6937, 0.7425, 0.7675, 0.7798],
            'FAJSCC (2026)':        [0.7091, 0.7592, 0.7879, 0.8035],
            'ST-JSCC (w/o LAM)':    [0.7128, 0.7841, 0.8214, 0.8404],
            'ST-JSCC (w/o RM)':     [0.6207, 0.7099, 0.7731, 0.8128],
            'ST-JSCC (w/o GFFN)':   [0.6673, 0.7421, 0.7840, 0.8063],
            'ST-JSCC (Full Model)': [0.7147, 0.7894, 0.8284, 0.8481],
        },
        "Kodak-AWGN-MS-SSIM": {
            'ResJSCC (2019)':       [0.8647, 0.9097, 0.9331, 0.9449],
            'FAJSCC (2026)':        [0.8833, 0.9219, 0.9432, 0.9544],
            'ST-JSCC (w/o LAM)':    [0.8628, 0.9154, 0.9432, 0.9576],
            'ST-JSCC (w/o RM)':     [0.8045, 0.8733, 0.9177, 0.9438],
            'ST-JSCC (w/o GFFN)':   [0.8412, 0.9000, 0.9323, 0.9493],
            'ST-JSCC (Full Model)': [0.8672, 0.9192, 0.9460, 0.9596],
        },
    }
    plot_2x3("Kodak", all_data, snr_points)


