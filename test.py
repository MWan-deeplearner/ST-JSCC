import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="hydra")
warnings.filterwarnings("ignore", message=".*Future Hydra versions.*")

from utils.test_utils import *
import logging
import hydra

logging.getLogger('fvcore.nn.jit_analysis').setLevel(logging.ERROR)


@hydra.main(version_base='1.1', config_path="configs", config_name='model_eval')
def main(cfg):
    logger = logging.getLogger(__name__)
    # make data_info
    result_dict = get_total_eval_dict_list(cfg, logger)
    print(result_dict)
    print("Params (M), Runtime (ms), Flops (G), Max Memory (MB), Model Memory (MB): ")
    print(f"{result_dict.get("params_M", 0)       :.2f} \n"
          f"{result_dict.get("runtime_ms", 0)     :.2f} \n"
          f"{result_dict.get("Flops_G", 0)        :.2f} \n"
          f"{result_dict.get("max_memory_MB", 0)  :.2f} \n"
          f"{result_dict.get("model_memory_MB", 0):.2f} \n")
    SNRs    = [item["SNR"]     for item in result_dict["PSNR_dB"]]
    PSNRs   = [item["PSNR"]    for item in result_dict["PSNR_dB"]]
    SSIMs   = [item["SSIM"]    for item in result_dict["SSIM"]]
    MSSSIMs = [item["MS-SSIM"] for item in result_dict["MS-SSIM"]]
    LPIPSs  = [item["LPIPS"]   for item in result_dict["LPIPS"]]
    print(
        f"Performance, SNR(s) = " + ' '.join(f"{snr}" for snr in SNRs),
        ", PSNR(s), SSIM(s), MS-SSIM(s) = "
    )
    print("\n".join(f"{  psnr:.2f}" for psnr   in PSNRs))
    print("\n".join(f"{  ssim:.4f}" for ssim   in SSIMs))
    print("\n".join(f"{msssim:.4f}" for msssim in MSSSIMs))
    print("\n".join(f"{ lpips:.4f}" for lpips  in LPIPSs))
    print(
        f"Performance, SNR(s) = " + ' '.join(f"{snr}" for snr in SNRs),
        ", PSNR(s), SSIM(s), MS-SSIM(s) = "
    )
    print(", ".join(f"{  psnr:.2f}" for psnr   in PSNRs))
    print(", ".join(f"{  ssim:.4f}" for ssim   in SSIMs))
    print(", ".join(f"{msssim:.4f}" for msssim in MSSSIMs))
    print(", ".join(f"{ lpips:.4f}" for lpips  in LPIPSs))

if __name__ == '__main__':
    main()
















