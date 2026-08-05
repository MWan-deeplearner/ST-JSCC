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

    dataset_list = ["Set5", "Set14"]
    channel_list = ["AWGN", "Rayleigh"]
    metrics_list = ["PSNR",  "SSIM", "MS-SSIM"]
    all_data = {
        "Set14-AWGN-PSNR": {
            'ConvJSCC (2019)':  [21.82, 22.58, 23.01, 23.23],
            'ResJSCC (2019)':   [23.69, 24.86, 25.54, 25.91],
            'SwinJSCC (2025)':  [23.77, 24.77, 25.36, 25.68],
            'LICRFJSCC (2025)': [23.30, 24.56, 25.39, 25.89],
            'LAJSCC (2026)':    [23.66, 24.83, 25.55, 25.96],
            'FAJSCC (2026)':    [23.94, 25.14, 25.92, 26.37],
            'ST-JSCC (Ours)':   [24.11, 25.79, 26.89, 27.54],
        },
        "Set14-AWGN-SSIM": {
            'ConvJSCC (2019)':  [0.5794, 0.6234, 0.6473, 0.6597],
            'ResJSCC (2019)':   [0.6511, 0.7041, 0.7319, 0.7459],
            'SwinJSCC (2025)':  [0.6451, 0.6953, 0.7231, 0.7379],
            'LICRFJSCC (2025)': [0.6321, 0.6945, 0.7345, 0.7577],
            'LAJSCC (2026)':    [0.6433, 0.6985, 0.7308, 0.7486],
            'FAJSCC (2026)':    [0.6569, 0.7130, 0.7466, 0.7653],
            'ST-JSCC (Ours)':   [0.6602, 0.7361, 0.7795, 0.8028],
        },
        "Set14-AWGN-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8368, 0.8733, 0.8923, 0.9019],
            'ResJSCC (2019)':   [0.8679, 0.9083, 0.9289, 0.9392],
            'SwinJSCC (2025)':  [0.8802, 0.9132, 0.9304, 0.9392],
            'LICRFJSCC (2025)': [0.8575, 0.9004, 0.9259, 0.9401],
            'LAJSCC (2026)':    [0.8652, 0.9051, 0.9273, 0.9392],
            'FAJSCC (2026)':    [0.8738, 0.9122, 0.9339, 0.9457],
            'ST-JSCC (Ours)':   [0.8667, 0.9138, 0.9391, 0.9521],
        },
        "Set14-Rayleigh-PSNR": {
            'ConvJSCC (2019)':  [21.03, 21.80, 22.20, 22.43],
            'ResJSCC (2019)':   [22.92, 23.90, 24.55, 24.99],
            'SwinJSCC (2025)':  [23.11, 23.92, 24.49, 24.88],
            'LICRFJSCC (2025)': [22.44, 23.44, 24.15, 24.63],
            'LAJSCC (2026)':    [22.85, 23.74, 24.39, 24.84],
            'FAJSCC (2026)':    [23.14, 24.11, 24.81, 25.29],
            'ST-JSCC (Ours)':   [23.25, 24.51, 25.46, 26.13],
        },
        "Set14-Rayleigh-SSIM": {
            'ConvJSCC (2019)':  [0.5646, 0.5956, 0.6169, 0.6306],
            'ResJSCC (2019)':   [0.6142, 0.6591, 0.6872, 0.7044],
            'SwinJSCC (2025)':  [0.6153, 0.6554, 0.6830, 0.7015],
            'LICRFJSCC (2025)': [0.5863, 0.6371, 0.6728, 0.6971],
            'LAJSCC (2026)':    [0.6069, 0.6492, 0.6785, 0.6984],
            'FAJSCC (2026)':    [0.6209, 0.6670, 0.6990, 0.7204],
            'ST-JSCC (Ours)':   [0.6261, 0.6861, 0.7261, 0.7501],
        },
        "Set14-Rayleigh-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8101, 0.8438, 0.8647, 0.8775],
            'ResJSCC (2019)':   [0.8428, 0.8810, 0.9037, 0.9174],
            'SwinJSCC (2025)':  [0.8616, 0.8906, 0.9091, 0.9208],
            'LICRFJSCC (2025)': [0.8318, 0.8711, 0.8970, 0.9138],
            'LAJSCC (2026)':    [0.8429, 0.8772, 0.8997, 0.9144],
            'FAJSCC (2026)':    [0.8514, 0.8868, 0.9093, 0.9236],
            'ST-JSCC (Ours)':   [0.8447, 0.8868, 0.9134, 0.9294],
        },
        "Set5-AWGN-PSNR": {
            'ConvJSCC (2019)':  [22.90, 23.94, 24.55, 24.90],
            'ResJSCC (2019)':   [25.19, 26.73, 27.70, 28.27],
            'SwinJSCC (2025)':  [25.26, 26.59, 27.43, 27.91],
            'LICRFJSCC (2025)': [24.26, 25.93, 27.12, 27.88],
            'LAJSCC (2026)':    [24.97, 26.57, 27.65, 28.32],
            'FAJSCC (2026)':    [25.36, 26.98, 28.12, 28.83],
            'ST-JSCC (Ours)':   [25.56, 27.47, 28.79, 29.62],
        },
        "Set5-AWGN-SSIM": {
            'ConvJSCC (2019)':  [0.5945, 0.6521, 0.6875, 0.7078],
            'ResJSCC (2019)':   [0.6967, 0.7616, 0.7980, 0.8176],
            'SwinJSCC (2025)':  [0.6899, 0.7508, 0.7858, 0.8049],
            'LICRFJSCC (2025)': [0.6547, 0.7294, 0.7792, 0.8096],
            'LAJSCC (2026)':    [0.6787, 0.7440, 0.7840, 0.8070],
            'FAJSCC (2026)':    [0.6929, 0.7590, 0.8010, 0.8253],
            'ST-JSCC (Ours)':   [0.6973, 0.7745, 0.8181, 0.8420],
        },
        "Set5-AWGN-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8648, 0.8995, 0.9177, 0.9270],
            'ResJSCC (2019)':   [0.8909, 0.9284, 0.9478, 0.9577],
            'SwinJSCC (2025)':  [0.8963, 0.9275, 0.9439, 0.9524],
            'LICRFJSCC (2025)': [0.8748, 0.9162, 0.9405, 0.9540],
            'LAJSCC (2026)':    [0.8823, 0.9209, 0.9422, 0.9537],
            'FAJSCC (2026)':    [0.8908, 0.9270, 0.9478, 0.9589],
            'ST-JSCC (Ours)':   [0.8950, 0.9341, 0.9538, 0.9637],
        },
        "Set5-Rayleigh-PSNR": {
            'ConvJSCC (2019)':  [21.02, 22.11, 22.68, 22.92],
            'ResJSCC (2019)':   [24.32, 25.60, 26.52, 27.16],
            'SwinJSCC (2025)':  [24.22, 25.28, 26.08, 26.66],
            'LICRFJSCC (2025)': [23.57, 24.83, 25.79, 26.49],
            'LAJSCC (2026)':    [24.08, 25.22, 26.12, 26.80],
            'FAJSCC (2026)':    [24.31, 25.57, 26.55, 27.28],
            'ST-JSCC (Ours)':   [24.49, 25.93, 27.10, 27.99],
        },
        "Set5-Rayleigh-SSIM": {
            'ConvJSCC (2019)':  [0.6268, 0.6548, 0.6786, 0.6928],
            'ResJSCC (2019)':   [0.6799, 0.7310, 0.7649, 0.7868],
            'SwinJSCC (2025)':  [0.6604, 0.7090, 0.7432, 0.7664],
            'LICRFJSCC (2025)': [0.6201, 0.6776, 0.7188, 0.7470],
            'LAJSCC (2026)':    [0.6496, 0.6981, 0.7327, 0.7568],
            'FAJSCC (2026)':    [0.6615, 0.7138, 0.7510, 0.7780],
            'ST-JSCC (Ours)':   [0.6786, 0.7407, 0.7835, 0.8101],
        },
        "Set5-Rayleigh-MS-SSIM": {
            'ConvJSCC (2019)':  [0.8368, 0.8689, 0.8868, 0.8967],
            'ResJSCC (2019)':   [0.8734, 0.9079, 0.9287, 0.9416],
            'SwinJSCC (2025)':  [0.8759, 0.9056, 0.9245, 0.9364],
            'LICRFJSCC (2025)': [0.8540, 0.8910, 0.9154, 0.9313],
            'LAJSCC (2026)':    [0.8636, 0.8961, 0.9177, 0.9319],
            'FAJSCC (2026)':    [0.8699, 0.9034, 0.9249, 0.9390],
            'ST-JSCC (Ours)':   [0.8772, 0.9126, 0.9350, 0.9484],
        }
    }
    for dataset in dataset_list:
        for channel in channel_list:
            for metrics in metrics_list:
                key = f"{dataset}-{channel}-{metrics}"
                if key in all_data.keys():
                    data = all_data[key]
                    plot_jscc_results(snr_points, data, dataset, channel, metrics)


