import matplotlib.pyplot as plt
import numpy as np

snr_all = np.arange(1, 11)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
psnr_all = np.array([26.16, 26.75, 27.27, 27.72, 28.11, 28.44, 28.71, 28.94, 29.14, 29.29])

selected_indices = [0, 3, 6, 9]
snr_discrete = snr_all[selected_indices]
psnr_discrete = psnr_all[selected_indices]

plt.figure(figsize=(8, 5.5), dpi=300)

plt.plot(
    snr_all,
    psnr_all,
    label='Full SNR Range (1–10 dB)',
    color='#1f77b4',
    linestyle='-',
    marker='o',
    markersize=6,
    linewidth=2
)

plt.plot(
    snr_discrete,
    psnr_discrete,
    label='Discrete Sampling (SNR = {1, 4, 7, 10} dB)',
    color='#d62728',
    linestyle='--',
    marker='s',
    markersize=7,
    linewidth=2
)

plt.title('PSNR Performance Comparison across SNR Regimes', fontsize=12, fontweight='bold')
plt.xlabel('Channel SNR (dB)', fontsize=11)
plt.ylabel('PSNR (dB)', fontsize=11)

plt.xticks(snr_all)
plt.xlim(0.5, 10.5)
plt.ylim(25.5, 30.0)

plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=10, loc='lower right')

plt.tight_layout()

plt.savefig('snr_sampling_comparison.pdf', dpi=300)
plt.show()