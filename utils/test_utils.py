import random
import os
import gc
import time
import pickle

from fvcore.nn import FlopCountAnalysis
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image
import lpips

from .loss_maker import get_loss_info, get_task_info, LossMaker
from .data_maker import DataMaker
from model.model_maker import ModelMaker


def set_model_info(cfg):
    get_loss_info(cfg)
    get_task_info(cfg)


def get_total_eval_dict_list(cfg, logger):
    set_model_info(cfg)
    SNR_info_list = cfg.SNR_info_list
    result_dict_name = get_result_dict_name(cfg)
    result_dict = load_result_dict(cfg, result_dict_name)
    if result_dict is not None:
        logger.info(f'Result_dict {result_dict_name} is loaded.')
        if not cfg.visual_data:
            return result_dict
    result_dict = {"PSNR_dB": [], "SSIM": [], "MS-SSIM": [], "LPIPS": []}
    runtime_list = []
    for i, SNR in enumerate(SNR_info_list):
        model_eval_dict = get_model_eval_dict(cfg, logger, SNR)
        if i == 0:
            result_dict["model_memory_MB"]   = model_eval_dict["Mmemory"]
            result_dict["params_M"]          = float(model_eval_dict["Mparams"])
            if cfg.test_data == "DIV2K":
                result_dict["Flops_G"]       = model_eval_dict["GFlops_1536x2048"]
                result_dict["max_memory_MB"] = model_eval_dict["max_memory_MB_1536x2048"]
            elif cfg.test_data == "Kodak":
                result_dict["Flops_G"]       = model_eval_dict["GFlops_512x768"]
                result_dict["max_memory_MB"] = model_eval_dict["max_memory_MB_512x768"]
        result_dict["PSNR_dB"].append({"SNR": SNR, "PSNR":    model_eval_dict["PSNR"]})
        result_dict["SSIM"].append(   {"SNR": SNR, "SSIM":    model_eval_dict["SSIM"]})
        result_dict["MS-SSIM"].append({"SNR": SNR, "MS-SSIM": model_eval_dict["MS-SSIM"]})
        result_dict["LPIPS"].append({"SNR": SNR, "LPIPS": model_eval_dict["LPIPS"]})
        runtime_list.append(model_eval_dict["ms/image"])
    result_dict["runtime_ms"] = sum(runtime_list) / len(runtime_list)
    save_result_dict(cfg, result_dict, result_dict_name)
    logger.info(f'Result_dict {result_dict_name} is saved.')
    return result_dict


def get_model_save_name(cfg):
    data = cfg.data_info.data_name
    model_name = cfg.model_name
    chan_type = cfg.chan_type
    metric = cfg.performance_metric
    rcpp = cfg.rcpp
    save_name = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_{metric}.pt"
    return save_name

def get_result_dict_name(cfg):
    model_name = cfg.model_name
    train_data = cfg.data_info.data_name
    test_data = cfg.test_data
    chan_type = cfg.chan_type
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    dict_name = (f"{model_name}_{train_data}_{test_data}_{chan_type}_"
                 f"rcpp{rcpp}.pkl")
    return dict_name    

def load_result_dict(cfg, dict_name):
    dict_folder = cfg.result_dicts_dir
    dict_path = os.path.join(dict_folder, dict_name)
    if os.path.exists(dict_path):
        with open(dict_path, "rb") as f:
            loaded_dict = pickle.load(f)
    else:
        loaded_dict = None
    return loaded_dict 
    
def save_result_dict(cfg, result_dict, dict_name):
    dict_folder = cfg.result_dicts_dir
    os.makedirs(dict_folder, exist_ok=True)
    dict_path = os.path.join(dict_folder, dict_name)
    with open(dict_path, "wb") as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
def get_loaded_model(cfg, logger):
    data = cfg.data_info.data_name
    model_name = cfg.model_name
    rcpp = cfg.rcpp
    get_loss_info(cfg)
    get_task_info(cfg)
    chan_type = cfg.chan_type
    cfg.rcpp = rcpp
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    save_dir = cfg.save_dir
    save_name = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_PSNR.pt"
    save_name_backup = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_PSNR_backup.pt"
    model_info_save_path = os.path.join(save_dir, save_name)
    model_backup_info_save_path = os.path.join(save_dir, save_name_backup)
    if not os.path.exists(model_info_save_path):
        logger.info(f'There is no trained model')
        logger.info(f'Model: {save_name} does not exist')
        return None
    model = ModelMaker(cfg)
    logger.info(f'Load model: {save_name}.')
    try:
        model.load_state_dict(torch.load(model_info_save_path))
        logger.info(f'The saved model is loaded')
    except Exception as ex:
        model.load_state_dict(torch.load(model_backup_info_save_path))
        logger.info(f'Error occured during saved model is loaded')
        logger.info(f'Error info:',ex)
        logger.info(f'The saved backup model is loaded')
    saved_model_epoch = model.epoch.item()
    logger.info(f'Loaded model trained epoch: {int(saved_model_epoch)}.')
    # random_seed_num = int(saved_model_epoch)
    # torch.manual_seed(random_seed_num)
    # np.random.seed(random_seed_num)
    # random.seed(random_seed_num)
    return model  
    
    
def get_specific_model_result_dict(cfg, logger, model, testloader, criterion, SNR):
    device = torch.device(cfg.device)
    model.to(device)
    since = time.time()
    evaluater = ModelEvaluater(cfg, SNR)
    evaluation_dictionary = evaluater.one_epoch_eval(cfg, logger, model, testloader, criterion)
    evaluation_dictionary = add_flops_and_max_memory(cfg, logger, evaluation_dictionary,model)
    Mmemory = cal_MB(cfg, logger, model)
    evaluation_dictionary['Mmemory'] = Mmemory
    Mparams = get_n_model_params(cfg, logger, model)
    evaluation_dictionary['Mparams'] = Mparams #number of parameters of the model
    logger.info(f'---------------------------------------------------------------')
    time_elapsed = time.time() - since
    logger.info(f'model {cfg.model_name} result dict is made in '
                f'{time_elapsed // 60:.0f}m { time_elapsed % 60:.0f}s')
    logger.info(f'---------------------------------------------------------------')
    #Important for stable gpu use    
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return evaluation_dictionary


def get_model_eval_dict(cfg, logger, SNR):
    #load model
    model = get_loaded_model(cfg, logger)
    # make data_info
    logger.info(f'Model evaluation is started.')
    data_info = DataMaker(cfg)
    # make criterion
    model.d = 4
    criterion = LossMaker(cfg)
    random_seed_num = cfg.random_seed
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)
    evaluation_dictionary = get_specific_model_result_dict(
        cfg, logger, model, data_info.testloader, criterion, SNR
    )
    return evaluation_dictionary


class ModelEvaluater:
    def __init__(self, cfg, SNR):
        device = torch.device(cfg.device)
        self.device = device
        self.task = cfg.task_name
        self.SNR = SNR

    def one_epoch_eval(self, cfg, logger, model, testloader, criterion):
        evaluation_dictionary = self.eval_task(cfg, logger, model, testloader, criterion)
        return evaluation_dictionary

    def eval_task(self,cfg, logger, model, testloader, criterion):
        device = torch.device(cfg.device)
        model.eval()
        evaluation_dictionary = {}
        evaluation_dictionary['task'] = cfg.task_name
        test_epoch_psnr = 0
        test_epoch_ssim = 0
        test_epoch_msssim = 0
        test_epoch_lpips = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        cfg.performance_metric = "PSNR"
        cal_psnr = LossMaker(cfg)
        cfg.performance_metric = "SSIM"
        cal_ssim = LossMaker(cfg)
        cfg.performance_metric = "MS-SSIM"
        cal_msssim = LossMaker(cfg)
        cal_lpips = lpips.LPIPS(net="alex").to(cfg.device)
        idx = 0
        for file_idx, (images, labels) in enumerate(testloader):
            batch = images.shape[0]
            count += batch
            images = images.to(device)
            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                images_hat = model(images, SNR_info=self.SNR)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                total_forward_time += (end_time - start_time)
            _, psnr = cal_psnr(images_hat, images)
            test_epoch_psnr += psnr.item() * batch
            _, ssim = cal_ssim(images_hat, images)
            test_epoch_ssim += ssim.item() * batch
            _, msssim = cal_msssim(images_hat, images)
            test_epoch_msssim += msssim.item() * batch
            lpips_val = cal_lpips(images_hat, images).mean()
            test_epoch_lpips += lpips_val
            idx += 1
            if cfg.visual_data and self.SNR == cfg.visual_data_snr:
                print(self.SNR, idx, psnr)
                path = os.path.join(cfg.visual_results_dir, f"{cfg.test_data}_{str(idx).zfill(2)}.png")
                save_img = ((images_hat + 1) / 2).clamp(0, 1)[0]
                torchvision.utils.save_image(save_img, path)
            
            # if self.SNR == 10:
            #     images_hat_cpu = images_hat.cpu()
            #     images_hat_normalized = (images_hat_cpu + 1) / 2
            #     save_image(
            #         images_hat_normalized, os.path.join(
            #             cfg.visual_results_dir, f'idx{file_idx}.png'
            #         ), nrow=4
            #     )
            #     if idx in [13, 19, 20]:
            #         print(idx, psnr)
                
        test_epoch_psnr   = test_epoch_psnr   / count
        test_epoch_ssim   = test_epoch_ssim   / count
        test_epoch_msssim = test_epoch_msssim / count
        test_epoch_lpips  = test_epoch_lpips / count
        avg_ms_per_image  = (total_forward_time / count) * 1000.0
        # logger.info(f'Test count per epoch: {count}')
        # logger.info(f'Test loss: {test_epoch_total_loss}')
        # logger.info(f'{performance_metric}: {test_epoch_performance}')
        # logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        evaluation_dictionary["PSNR"]    = test_epoch_psnr
        evaluation_dictionary["SSIM"]    = test_epoch_ssim
        evaluation_dictionary["MS-SSIM"] = test_epoch_msssim
        evaluation_dictionary["LPIPS"]   = test_epoch_lpips
        evaluation_dictionary['ms/image'] = avg_ms_per_image
        return evaluation_dictionary


def add_flops_and_max_memory(cfg, logger, evaluation_dictionary, model):
    resolution_list = [(512,768),(1536,2048)]
    for H, W in resolution_list:
        GFlops, max_memory = cal_flops_and_memory(cfg, logger, model, H, W)
        key = f"{H}x{W}"
        evaluation_dictionary[f"GFlops_{key}"] = float(GFlops)
        evaluation_dictionary[f"max_memory_MB_{key}"] = float(max_memory)
    return evaluation_dictionary


def cal_flops_and_memory(cfg, logger, model, H, W):
    device = torch.device(cfg.device)
    # logger.info(f'Input resolution: {H} x {W}')
    input_image = torch.rand(1, 3, H, W).float().to(device)
    model = model.to(device)
    model.eval()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        # ---------------- FLOPs ----------------
        flops = FlopCountAnalysis(model, input_image)
        GFlops = flops.total() / 1e9
        # ---------------- Memory ----------------
        _ = model(input_image)
        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            max_memory = 0.0
    # logger.info(f'GFlops: {GFlops:.4f}')
    # logger.info(f'Max GPU Memory: {max_memory:.2f} MB')
    return GFlops, max_memory


def get_n_model_params(cfg, logger, model):
    model.to('cpu')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    Mparams = params / 10**6
    # logger.info(f'The number of parameters of model: {Mparams} M')
    return Mparams


def cal_MB(cfg, logger, model):
    device = torch.device(cfg.device)
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    previous_memory = to_MB(torch.cuda.memory_allocated())
    model.to(device)
    model.eval()
    Mmemory = to_MB(torch.cuda.memory_allocated())
    Model_Memory = Mmemory-previous_memory
    # logger.info(f'Allocated gpu memory for model usage: {Model_Memory} Mb')
    return Model_Memory


def to_MB(a):
    return a/1024.0/1024.0
    