from utils.loss_maker import *
from utils.optimizer_maker import *
from utils.train_utils import *
from model.model_maker import *
import random
import os
import torch.optim as optim
import torch
import numpy as np
import logging
import hydra
from omegaconf import DictConfig


def load_trained_model(cfg, logger, model):
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    model_name = cfg.model_name
    save_dir = cfg.save_dir

    save_name = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_{metric}.pt"
    save_name_backup = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_{metric}_backup.pt"
    model_info_save_path = os.path.join(save_dir, save_name)
    model_backup_info_save_path = os.path.join(save_dir, save_name_backup)
    if os.path.exists(model_info_save_path):
        try:
            # model.load_state_dict(torch.load(model_info_save_path))
            ckpt = torch.load(model_info_save_path, map_location="cpu")
            model.load_state_dict(ckpt)
            logger.info(f'The saved model is loaded')
            saved_model_epoch = model.epoch.item()
            logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')

        except Exception as ex:
            logger.info(f'Error occured during saved model is loaded')
            logger.info(f'Error info:', ex)
            try:
                ckpt = torch.load(model_backup_info_save_path, map_location="cpu")
                model.load_state_dict(ckpt)
                # model.load_state_dict(torch.load(model_backup_info_save_path))
                logger.info(f'The saved backup model is loaded')
                saved_model_epoch = model.epoch.item()
                logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')
            except Exception as e:
                logger.info(f'Error occured during backup model is loaded')
                logger.info(f'Error info:', e)

                logger.info(f'Train epoch is initialized, new default model is made')
                model = ModelMaker(cfg)  # make model and set appropriate task name
    else:
        logger.info(f'There is no trained model')

    return model


@hydra.main(version_base='1.1', config_path="configs", config_name='train')
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)

    device = cfg.device
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'device: {device}')

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # set random seed number
    random_seed_num = cfg.random_seed
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)

    # make data_info
    data_info = DataMaker(cfg)

    # make model
    model = ModelMaker(cfg)  # make model
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    # make criterion
    criterion = LossMaker(cfg)

    logger.info(f'---' * 10)
    logger.info(f'Try loading pretrained model')
    model = load_trained_model(cfg, logger, model)
    logger.info(f'---' * 10)

    saved_model_epoch = model.epoch.item()

    model = model.to(device)

    should_stop = saved_model_epoch >= cfg.total_max_epoch
    if should_stop:
        logger.info(f"saved model already exists, total_max_epoch is {cfg.total_max_epoch}")
        return None

    random_seed_num = int(saved_model_epoch)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)

    # make optimizer
    base_model = model
    optimizer = OptimizerMaker(base_model, cfg)

    # make scheduler
    milestones = [99999999]  # [60, 80]
    if cfg.data_info.data_name in ["Flickr30k"]:
        milestones = [70 - saved_model_epoch, 75 - saved_model_epoch]  # milestones = [90,95]

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

    logger.info(hydra_cfg['runtime']['output_dir'])
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'Task: {cfg.task_name}')
    logger.info(f'Data: {cfg.data_info.data_name}')
    logger.info(f'chan_type: {cfg.chan_type}')
    logger.info(f'rcpp: {cfg.rcpp}')
    logger.info(f'performance_metric: {cfg.performance_metric}')
    logger.info(f'Model: {cfg.model_name}')
    logger.info(f'Learning rate: {cfg.learning_rate}')

    # train model
    train_model(cfg, logger, model, data_info, criterion, optimizer, scheduler)


if __name__ == '__main__':
    main()




