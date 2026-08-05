import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

methods_config = {
	'$\\alpha=0.1$':           {
		'color': '#2F4F4F', 'marker': 'o', 'linewidth': 1.5, 'zorder': 5
	},
	'$\\alpha=0.5$':           {
		'color': '#4682B4', 'marker': '^', 'linewidth': 1.5, 'zorder': 5
	},
	'$\\alpha=1.0$ (default)': {
		'color': '#FF0000', 'marker': 's', 'linewidth': 2.5, 'zorder': 10
	},
	'$\\alpha=1.5$':            {
		'color': '#D2691E', 'marker': 'p', 'linewidth': 1.5, 'zorder': 5
	},
	'$\\alpha=2.0$':           {
		'color': '#9ACD32', 'marker': 'D', 'linewidth': 1.5, 'zorder': 5
	},
}

markersize = 7
label_size = 14
tick_sizes = 14
legend_size = 14
title_size = 14
grid_alpha = 0.7


def plot_psnr_curves(snr_list, results_dict, save_name='psnr_curves.pdf'):
	plt.figure(figsize=(8, 6), dpi=300)
	
	for method, psnr_values in results_dict.items():
		if method in methods_config:
			cfg = methods_config[method]
			plt.plot(snr_list, psnr_values,
					 label=method,
					 color=cfg['color'],
					 marker=cfg['marker'],
					 linewidth=cfg['linewidth'],
					 markersize=markersize,
					 zorder=cfg['zorder'])
		else:
			# 如果方法名不在配置中，自动分配一个颜色和标记（这里简单处理）
			plt.plot(snr_list, psnr_values, label=method, linewidth=1.5, markersize=markersize)
	
	plt.xlabel('SNR (dB)', fontsize=label_size)
	plt.ylabel('PSNR (dB)$\\uparrow$', fontsize=label_size)
	plt.xticks(fontsize=tick_sizes, ticks=snr_list)
	plt.yticks(fontsize=tick_sizes)
	plt.grid(True, linestyle='--', alpha=grid_alpha)
	plt.legend(loc='lower right', fontsize=legend_size, frameon=True)
	plt.tight_layout()
	plt.title('Ablation study on hyperparameter $\\alpha$ in ELU function', fontsize=title_size)
	plt.savefig(save_name, bbox_inches='tight')


if __name__ == "__main__":
	snr_points = [1, 4, 7, 10]

	example_data = {
		'$\\alpha=0.1$':           [26.00, 27.79, 28.60, 28.97],
		'$\\alpha=0.5$':           [25.82, 27.42, 28.53, 29.12],
		'$\\alpha=1.0$ (default)': [26.12, 27.75, 28.77, 29.36],
		'$\\alpha=1.5$':           [25.95, 27.70, 28.76, 29.27],
		'$\\alpha=2.0$':           [26.21, 27.67, 28.68, 29.39],
	}

	plot_psnr_curves(snr_points, example_data, save_name='my_5methods_psnr.pdf')