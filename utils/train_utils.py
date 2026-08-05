import os
import time
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

from utils.data_maker import *
from utils import *


def save_loss_plot(total_loss_info_list, cfg, logger):
    # get plot info
    epoch_list = [1 + i for i in range(len(total_loss_info_list[1]))]

    # plt.rcParams.update({'text.usetex': True})
    # plt.rcParams["figure.figsize"] = (14,8)
    fig, ax1 = plt.subplots()

    lines = ax1.plot(epoch_list, total_loss_info_list[1], label="train loss")

    plt.title(f"loss per epoch")
    ax1.set_xlabel(r'Epochs', fontsize=18)
    ax1.set_ylabel(f'{total_loss_info_list[0]}', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout(rect=[0, 0, 0.6, 0.8])

    # save results
    save_dir = cfg.loss_curve_dir
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    model_name = cfg.model_name
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_name = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_{metric}.pdf"

    plot_save_name = os.path.join(save_dir, save_name)
    if save_name:
        logger.info(f'plot of loss info is saved')
        plt.savefig(plot_save_name)
    # plt.show()
    plt.clf()

    list_save_name = save_name + ".pkl"
    with open(save_dir + list_save_name, "wb") as f:
        logger.info(f'list of loss info is saved')
        pickle.dump(total_loss_info_list, f)

    # with open("save_dir+list_save_name","rb") as f:
    # total_loss_info_list = pickle.load(f)


def train_model(cfg: DictConfig, logger, model, data_info, criterion, optimizer, scheduler=None):
    total_max_epoch = cfg.total_max_epoch
    base_model = model
    saved_model_epoch = int(base_model.epoch.item())
    since = time.time()
    trainer = Trainer(cfg)

    # save information for loss curve
    total_loss_info_list = [cfg.loss_name]
    train_loss_list = []

    # save inforamtion for best performance model
    data = cfg.data_info.data_name
    chan_type = cfg.chan_type
    rcpp = str(cfg.rcpp).zfill(3)
    metric = cfg.performance_metric
    model_name = cfg.model_name
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    save_name = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_{metric}.pt"
    save_name_backup = f"{model_name}_{data}_{chan_type}_rcpp{rcpp}_{metric}_backup.pt"

    criterion.current_epoch = base_model.epoch.item()

    # logger.info(f'model save epoch point: {save_point+1}')

    for epoch in range(saved_model_epoch, total_max_epoch):
        saved_model_epoch = base_model.epoch.item()
        logger.info(f'---------------------------------------------------------------')
        logger.info(f'Epoch {epoch + 1}/{total_max_epoch}')
        logger.info(f'loaded_model_trained_epoch: {saved_model_epoch}')

        seed = int(cfg.random_seed) * 20 + epoch
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # data_info.set_epoch(epoch=int(epoch))
        train_epoch_loss = trainer.one_epoch_train(
            cfg, logger, model, data_info.trainloader, criterion, optimizer
        )
        if scheduler:
            scheduler.step()

        base_model.add_epoch()

        train_loss_list.append(train_epoch_loss)
        since1 = time.time()
        try:
            torch.save(base_model.state_dict(), os.path.join(save_dir, save_name))
            saved_model_epoch = base_model.epoch.item()
            logger.info(f'saved_model_total_epoch: {saved_model_epoch}')
            logger.info(f'The model is saved at train epoch {epoch + 1}')
            # torch.save(base_model.state_dict(), os.path.join(save_dir, save_name_backup))
            # logger.info(f'backup model is saved at train epoch {epoch + 1}')
            # logger.info(f'One epoch train is finished')
        except Exception as ex:
            logger.info(f'Error occured during model save')
            logger.info(f'Error info:', ex)
            try:
                os.remove(os.path.join(save_dir, save_name))
            except OSError:
                pass
            torch.save(base_model.state_dict(), os.path.join(save_dir, save_name))
            saved_model_epoch = base_model.epoch.item()
            logger.info(f'saved_model_total_epoch: {saved_model_epoch}')
            logger.info(f'The model is saved at train epoch {epoch + 1}')
            # try:
            #     torch.save(base_model.state_dict(), os.path.join(save_dir, save_name_backup))
            #     logger.info(f'backup model is saved at train epoch {epoch + 1}')
            #     logger.info(f'One epoch train is finished')
            # except:
            #     try:
            #         os.remove(save_dir + save_name_backup)
            #     except OSError:
            #         pass
            #     torch.save(base_model.state_dict(), os.path.join(save_dir, save_name_backup))
            #     logger.info(f'backup model is saved at train epoch {epoch + 1}')
            #     logger.info(f'One epoch train is finished')
        time_elapsed = time.time() - since1
        logger.info(f'Model save complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    total_loss_info_list.append(train_loss_list)
    save_loss_plot(total_loss_info_list, cfg, logger)
    time_elapsed = time.time() - since
    logger.info(f'---------------------------------------------------------------')
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return None


class Trainer:

    def __init__(self, cfg: DictConfig):
        device = torch.device(cfg.device)
        self.device = device
        self.task = cfg.task_name

    def one_epoch_train(self, cfg, logger, model, trainloader, criterion, optimizer):
        if self.task == "ITrandomSNR":
            train_epoch_loss = self.train_ITrandomSNR_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        elif self.task == "FAITrandomSNR":
            train_epoch_loss = self.train_FAITrandomSNR_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        elif self.task == "ITrandomSNR_freq":
            train_epoch_loss = self.train_ITrandomSNR_freq_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        elif self.task == "ITrandomSNR_Gaussian":
            train_epoch_loss = self.train_ITrandomSNR_Gaussian_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        elif self.task == "ITrandomSNR_GAN":
            train_epoch_loss = self.train_ITrandomSNR_GAN_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        elif self.task == "ITrandomSNR_Separate":
            train_epoch_loss = self.train_ITrandomSNR_Separate_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        elif self.task == "ITrandomSNR_Encoder":
            train_epoch_loss = self.train_ITrandomSNR_Encoder_task(
                cfg, logger, model, trainloader, criterion, optimizer
            )
        else:
            raise ValueError(f'{self.task} task train is not implemented yet')
        return train_epoch_loss

    def train_ITrandomSNR_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            images_hat = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images)
            total_loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return train_epoch_total_loss

    def train_ITrandomSNR_Encoder_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            encoder_output, decoder_input = model.small_forward(images, SNR_info=SNR_info)
            loss = criterion(decoder_input, encoder_output.detach())[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start_time = time.time()
            images_hat = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return train_epoch_total_loss

    def train_ITrandomSNR_Separate_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            images_hat, y, y_hat = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images)
            total_loss = total_loss + criterion(y_hat, y)[0]
            total_loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return train_epoch_total_loss

    def train_ITrandomSNR_GAN_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            images_hat, loss_sub, d_loss, g_adv = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images)
            optimizer.zero_grad()
            for p in model.decoder.discriminator.parameters():
                p.requires_grad = False
            loss = total_loss + cfg.alpha * loss_sub + cfg.beta * g_adv
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
            for p in model.decoder.discriminator.parameters():
                p.requires_grad = True
            images_hat, loss_sub, d_loss, g_adv = model(images, SNR_info=SNR_info)
            loss = d_loss
            loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return train_epoch_total_loss

    def train_ITrandomSNR_freq_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            images_hat, b1, b2 = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images, b1, model.decoder.f, model.decoder.f_inv)
            total_loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return train_epoch_total_loss

    def train_ITrandomSNR_Gaussian_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            images_hat, p_m, p_s, q_m, q_s = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images, p_m, p_s, q_m, q_s)
            total_loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        return train_epoch_total_loss

    def train_FAITrandomSNR_task(
            self, cfg: DictConfig, logger, model, trainloader, criterion, optimizer
    ):
        device = self.device
        since = time.time()
        model.train()
        train_epoch_total_loss = 0
        train_epoch_performance = 0
        performance_metric = cfg.performance_metric
        count = 0
        total_forward_time = 0.0
        for images, labels in trainloader:
            count += images.shape[0]
            images = images.to(device)
            optimizer.zero_grad()
            SNR_info_list = cfg.SNR_info_list
            SNR_info = random.choice(SNR_info_list)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            images_hat, decision = model(images, SNR_info=SNR_info)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            total_forward_time += (end_time - start_time)
            total_loss, performance = criterion(images_hat, images, decision)
            total_loss.backward()
            optimizer.step()
            train_epoch_total_loss += total_loss.item() * images.size(0)
            train_epoch_performance += performance.item() * images.size(0)
            if count % 1000 == 0:
                logger.info(f'current count: {count}')
        train_epoch_total_loss = train_epoch_total_loss / count
        train_epoch_performance = train_epoch_performance / count
        avg_ms_per_image = (total_forward_time / count) * 1000.0
        logger.info(f'train count per epoch: {count}')
        logger.info(f'Train loss: {train_epoch_total_loss}')
        logger.info(f'{performance_metric}: {train_epoch_performance}')
        logger.info(f'Forward time: {avg_ms_per_image:.4f} ms/image')
        time_elapsed = time.time() - since
        logger.info(f'Training epoch complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        return train_epoch_total_loss








