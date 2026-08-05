import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
methods_config = {
    'ConvJSCC (2019)':  {'color': '#2F4F4F', 'marker': 'o', 'linewidth': 1.5, 'zorder':  5},
    'ResJSCC (2019)':   {'color': '#4682B4', 'marker': '^', 'linewidth': 1.5, 'zorder':  5},
    'SwinJSCC (2025)':  {'color': '#00CED1', 'marker': 'p', 'linewidth': 1.5, 'zorder':  5},
    'LICRFJSCC (2025)': {'color': '#9370DB', 'marker': '*', 'linewidth': 1.5, 'zorder':  5},
    'LAJSCC (2026)':    {'color': '#9ACD32', 'marker': 'D', 'linewidth': 1.5, 'zorder':  5},
    'FAJSCC (2026)':    {'color': '#D2691E', 'marker': 'v', 'linewidth': 1.5, 'zorder':  5},
    'ST-JSCC (Ours)':   {'color': '#FF0000', 'marker': 's', 'linewidth': 2.5, 'zorder': 10},
}
markersize  =  7
label_size  = 14
tick_sizes  = 14
legend_size = 14
title_size  = 14
grid_alpha  = 0.7

def plot_jscc_results(snr_list, results_dict, dataset, channel, metrics):
    plt.figure(figsize=(8, 6), dpi=300)
    for method, psnr_values in results_dict.items():
        if method in methods_config:
            cfg = methods_config[method]
            plt.plot(snr_list, psnr_values, label=method,
                     color=cfg['color'], marker=cfg['marker'],
                     linewidth=cfg['linewidth'], markersize=markersize,
                     zorder=cfg['zorder'])
    plt.xlabel('SNR (dB)', fontsize=label_size)
    if   metrics == "PSNR":
        plt.ylabel('PSNR (dB)$\\uparrow$', fontsize=label_size)
    elif metrics == "SSIM":
        plt.ylabel('SSIM$\\uparrow$',      fontsize=label_size)
    elif metrics == "MS-SSIM":
        plt.ylabel('MS-SSIM$\\uparrow$',   fontsize=label_size)
    plt.xticks(fontsize=tick_sizes, ticks=snr_list)
    plt.yticks(fontsize=tick_sizes)
    plt.grid(True, linestyle='--', alpha=grid_alpha)
    plt.legend(loc='lower right', fontsize=legend_size, frameon=True)
    plt.tight_layout()
    plt.title(f"Testing on {dataset} dataset, {channel} channel, {metrics} metrics.",
              fontsize=title_size)
    plt.savefig(f'quantitative_{dataset}_{channel}_{metrics}.pdf', bbox_inches='tight')
    # plt.show()


if __name__ == "__main__":
    snr_points = [1, 4, 7, 10]

    dataset_list = ["DIV2K", "Kodak"]
    # dataset_list = ["Kodak"]
    channel_list = ["AWGN",  "Rayleigh"]
    # channel_list = ["AWGN"]
    metrics_list = ["PSNR",  "SSIM", "MS-SSIM"]
    # metrics_list = ["LPIPS"]
    all_data = {
        "DIV2K-AWGN-PSNR": {
            'ConvJSCC (2019)':  [23.00, 23.90, 24.40, 24.67],
            'ResJSCC (2019)':   [25.82, 27.02, 27.73, 28.12],
            'SwinJSCC (2025)':  [25.85, 26.74, 27.24, 27.51],
            'LICRFJSCC (2025)': [25.14, 26.24, 26.94, 27.35],
            'LAJSCC (2026)':    [25.95, 27.13, 27.86, 28.27],
            'FAJSCC (2026)':    [26.30, 27.53, 28.30, 28.75],
            'GLJSCC (Ours)':    [26.21, 27.97, 29.08, 29.74],
        },
        "DIV2K-AWGN-SSIM": {
            'ConvJSCC (2019)':  [0.6135, 0.6594, 0.6827, 0.6941],
            'ResJSCC (2019)':   [0.7234, 0.7740, 0.7997, 0.8126],
            'SwinJSCC (2025)':  [0.7123, 0.7597, 0.7853, 0.7987],
            'LICRFJSCC (2025)': [0.7050, 0.7621, 0.7968, 0.8163],
            'LAJSCC (2026)':    [0.7163, 0.7694, 0.7991, 0.8149],
            'FAJSCC (2026)':    [0.7312, 0.7849, 0.8153, 0.8317],
            'GLJSCC (Ours)':    [0.7218, 0.7969, 0.8351, 0.8541],
        },
        "DIV2K-AWGN-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8396, 0.8822, 0.9037, 0.9144],
            'ResJSCC (2019)':   [0.8830, 0.9237, 0.9442, 0.9546],
            'SwinJSCC (2025)':  [0.9002, 0.9308, 0.9465, 0.9544],
            'LICRFJSCC (2025)': [0.8803, 0.9199, 0.9426, 0.9549],
            'LAJSCC (2026)':    [0.8823, 0.9223, 0.9438, 0.9550],
            'FAJSCC (2026)':    [0.8936, 0.9304, 0.9501, 0.9603],
            'GLJSCC (Ours)':    [0.8778, 0.9259, 0.9500, 0.9620],
        },
        "DIV2K-Rayleigh-PSNR": {
            'ConvJSCC (2019)':  [22.71, 23.23, 23.53, 23.78],
            'ResJSCC (2019)':   [25.10, 26.02, 26.67, 27.12],
            'SwinJSCC (2025)':  [25.22, 25.95, 26.46, 26.81],
            'LICRFJSCC (2025)': [24.76, 25.68, 26.31, 26.73],
            'LAJSCC (2026)':    [25.17, 26.03, 26.64, 27.06],
            'FAJSCC (2026)':    [25.51, 26.42, 27.09, 27.55],
            'GLJSCC (Ours)':    [25.45, 26.71, 27.65, 28.30],
        },
        "DIV2K-Rayleigh-SSIM": {
            'ConvJSCC (2019)':  [0.6361, 0.6602, 0.6745, 0.6851],
            'ResJSCC (2019)':   [0.7005, 0.7392, 0.7639, 0.7799],
            'SwinJSCC (2025)':  [0.6976, 0.7343, 0.7593, 0.7757],
            'LICRFJSCC (2025)': [0.6720, 0.7208, 0.7537, 0.7752],
            'LAJSCC (2026)':    [0.6960, 0.7350, 0.7618, 0.7800],
            'FAJSCC (2026)':    [0.7044, 0.7460, 0.7748, 0.7940],
            'GLJSCC (Ours)':    [0.6971, 0.7548, 0.7920, 0.8142],
        },
        "DIV2K-Rayleigh-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8300, 0.8623, 0.8791, 0.8921],
            'ResJSCC (2019)':   [0.8660, 0.9012, 0.9229, 0.9365],
            'SwinJSCC (2025)':  [0.8907, 0.9159, 0.9319, 0.9420],
            'LICRFJSCC (2025)': [0.8582, 0.8955, 0.9195, 0.9346],
            'LAJSCC (2026)':    [0.8709, 0.9021, 0.9225, 0.9359],
            'FAJSCC (2026)':    [0.8797, 0.9100, 0.9296, 0.9422],
            'GLJSCC (Ours)':    [0.8588, 0.9010, 0.9274, 0.9432],
        },
        "Kodak-AWGN-PSNR": {
            'ConvJSCC (2019)':  [22.87, 23.69, 24.14, 24.37],
            'ResJSCC (2019)':   [25.28, 26.28, 26.83, 27.13],
            'SwinJSCC (2025)':  [25.51, 26.22, 26.61, 26.82],
            'LICRFJSCC (2025)': [25.35, 26.38, 27.01, 27.37],
            'LAJSCC (2026)':    [25.59, 26.54, 27.09, 27.40],
            'FAJSCC (2026)':    [25.91, 26.90, 27.49, 27.82],
            'GLJSCC (Ours)':    [26.12, 27.75, 28.77, 29.36],
        },
        "Kodak-AWGN-SSIM": {
            'ConvJSCC (2019)':  [0.6235, 0.6717, 0.6963, 0.7082],
            'ResJSCC (2019)':   [0.6937, 0.7425, 0.7675, 0.7798],
            'SwinJSCC (2025)':  [0.6918, 0.7366, 0.7611, 0.7740],
            'LICRFJSCC (2025)': [0.6992, 0.7542, 0.7875, 0.8061],
            'LAJSCC (2026)':    [0.6942, 0.7438, 0.7720, 0.7871],
            'FAJSCC (2026)':    [0.7091, 0.7592, 0.7879, 0.8035],
            'GLJSCC (Ours)':    [0.7147, 0.7894, 0.8284, 0.8481],
        },
        "Kodak-AWGN-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8283, 0.8777, 0.9035, 0.9163],
            'ResJSCC (2019)':   [0.8647, 0.9097, 0.9331, 0.9449],
            'SwinJSCC (2025)':  [0.8892, 0.9218, 0.9388, 0.9474],
            'LICRFJSCC (2025)': [0.8712, 0.9141, 0.9388, 0.9521],
            'LAJSCC (2026)':    [0.8728, 0.9142, 0.9369, 0.9489],
            'FAJSCC (2026)':    [0.8833, 0.9219, 0.9432, 0.9544],
            'GLJSCC (Ours)':    [0.8672, 0.9192, 0.9460, 0.9596],
        },
        "Kodak-AWGN-LPIPS": {
            'ConvJSCC (2019)':  [0.6140, 0.5268, 0.4709, 0.4414],
            'ResJSCC (2019)':   [0.5172, 0.4258, 0.3669, 0.3355],
            'SwinJSCC (2025)':  [0.4351, 0.3620, 0.3133, 0.2851],
            'LICRFJSCC (2025)': [0.5276, 0.4371, 0.3698, 0.3258],
            'LAJSCC (2026)':    [0.5331, 0.4456, 0.3833, 0.3436],
            'FAJSCC (2026)':    [0.4653, 0.3856, 0.3301, 0.2955],
            'ST-JSCC (Ours)':   [0.3408, 0.2401, 0.1840, 0.1582],
        },
        "Kodak-Rayleigh-PSNR": {
            'ConvJSCC (2019)':  [22.73, 22.95, 23.08, 23.25],
            'ResJSCC (2019)':   [24.58, 25.33, 25.85, 26.20],
            'SwinJSCC (2025)':  [25.03, 25.64, 26.06, 26.32],
            'LICRFJSCC (2025)': [24.52, 25.31, 25.86, 26.22],
            'LAJSCC (2026)':    [24.88, 25.59, 26.08, 26.41],
            'FAJSCC (2026)':    [25.21, 25.96, 26.50, 26.87],
            'GLJSCC (Ours)':    [25.31, 26.44, 27.26, 27.82],
        },
        "Kodak-Rayleigh-SSIM": {
            'ConvJSCC (2019)':  [0.6191, 0.6448, 0.6631, 0.6766],
            'ResJSCC (2019)':   [0.6641, 0.7013, 0.7258, 0.7416],
            'SwinJSCC (2025)':  [0.6719, 0.7065, 0.7305, 0.7465],
            'LICRFJSCC (2025)': [0.6557, 0.7023, 0.7339, 0.7546],
            'LAJSCC (2026)':    [0.6683, 0.7045, 0.7298, 0.7472],
            'FAJSCC (2026)':    [0.6798, 0.7181, 0.7450, 0.7633],
            'GLJSCC (Ours)':    [0.6802, 0.7384, 0.7761, 0.7981],
        },
        "Kodak-Rayleigh-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8058, 0.8412, 0.8671, 0.8852],
            'ResJSCC (2019)':   [0.8458, 0.8842, 0.9088, 0.9244],
            'SwinJSCC (2025)':  [0.8769, 0.9044, 0.9222, 0.9336],
            'LICRFJSCC (2025)': [0.8441, 0.8848, 0.9113, 0.9282],
            'LAJSCC (2026)':    [0.8562, 0.8894, 0.9116, 0.9263],
            'FAJSCC (2026)':    [0.8656, 0.8979, 0.9194, 0.9335],
            'GLJSCC (Ours)':    [0.8477, 0.8923, 0.9205, 0.9374],
        }
    }
    for dataset in dataset_list:
        for channel in channel_list:
            for metrics in metrics_list:
                key = f"{dataset}-{channel}-{metrics}"
                if key in all_data.keys():
                    data = all_data[key]
                    plot_jscc_results(snr_points, data, dataset, channel, metrics)


