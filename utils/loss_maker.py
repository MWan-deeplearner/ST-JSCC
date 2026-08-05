from torch import nn

from .utils import *
from .torch_msssim import ssim_, ms_ssim_


def get_task_info(cfg):
	if cfg.model_name == "FAJSCC":
		cfg.task_name = "FAITrandomSNR"
	elif cfg.model_name[:8] == "FreqJSCC":
		cfg.task_name = "ITrandomSNR_freq"
	elif cfg.model_name[:len("GaussianJSCC")] == "GaussianJSCC":
		cfg.task_name = "ITrandomSNR_Gaussian"
	elif cfg.model_name[:len("GANJSCC")] == "GANJSCC":
		cfg.task_name = "ITrandomSNR_GAN"
	elif cfg.model_name[:len("SeparateSCC")] == "SeparateSCC":
		cfg.task_name = "ITrandomSNR_Separate"
	elif cfg.model_name[:len("DilationJSCCV11")] == "DilationJSCCV11":
		cfg.task_name = "ITrandomSNR_Encoder"
	else:
		cfg.task_name = "ITrandomSNR"


def get_loss_info(cfg):
	cfg.loss_name = None
	get_task_info(cfg)

	if cfg.task_name == "FAITrandomSNR":
		if cfg.performance_metric == "PSNR":
			cfg.loss_name = "FAIT_MSE"
		elif cfg.performance_metric == "SSIM":
			cfg.loss_name = "FAIT_SSIM"
		elif cfg.performance_metric == "MS-SSIM":
			cfg.loss_name = "FAIT_MS-SSIM"
		else:
			raise ValueError(f'loss function for {cfg.performance_metric} of '
							 f'{cfg.task_name} task is not implemented yet')
	elif cfg.task_name == "ITrandomSNR_freq":
		if cfg.performance_metric == "PSNR":
			cfg.loss_name = "Freq_MSE"
		# elif cfg.performance_metric == "SSIM":
		# 	cfg.loss_name = "IT_SSIM"
		# elif cfg.performance_metric == "MS-SSIM":
		# 	cfg.loss_name = "IT_MS-SSIM"
		else:
			raise ValueError(f'loss function for {cfg.performance_metric} of '
							 f'{cfg.task_name} task is not implemented yet')
	elif cfg.task_name == "ITrandomSNR_Gaussian":
		if cfg.performance_metric == "PSNR":
			if cfg.data_mode == "test":
				cfg.loss_name = "IT_MSE"
			else:
				cfg.loss_name = "Gaussian_MSE"
		elif cfg.performance_metric == "SSIM":
			cfg.loss_name = "IT_SSIM"
		elif cfg.performance_metric == "MS-SSIM":
			cfg.loss_name = "IT_MS-SSIM"
		else:
			raise ValueError(f'loss function for {cfg.performance_metric} of '
							 f'{cfg.task_name} task is not implemented yet')
	else:
		if cfg.performance_metric == "PSNR":
			cfg.loss_name = "IT_MSE"
		elif cfg.performance_metric == "SSIM":
			cfg.loss_name = "IT_SSIM"
		elif cfg.performance_metric == "MS-SSIM":
			cfg.loss_name = "IT_MS-SSIM"
		else:
			raise ValueError(f'loss function for {cfg.performance_metric} of '
							 f'{cfg.task_name} task is not implemented yet')


def LossMaker(cfg): #cfg: DictConfig
	get_loss_info(cfg)
	if cfg.loss_name == "IT_MSE":
		loss = IT_MSE(cfg)
	elif cfg.loss_name == "IT_SSIM":
		loss = IT_SSIM(cfg)
	elif cfg.loss_name == "FAIT_MSE":
		loss = FAIT_MSE(cfg)
	elif cfg.loss_name == "Freq_MSE":
		loss = Freq_MSE(cfg)
	elif cfg.loss_name == "Gaussian_MSE":
		loss = Gaussian_MSE(cfg)
	elif cfg.loss_name == "FAIT_SSIM":
		loss = FAIT_SSIM(cfg)
	elif cfg.loss_name in ["IT_MS-SSIM","FAIT_MS-SSIM"]:
		loss = IT_MSSSIM(cfg)
	else:
		raise ValueError(f'{cfg.loss_name} is not implemented yet')
	return loss


class IT_MSE(torch.nn.Module):
	def __init__(self, cfg):
		super(IT_MSE, self).__init__()
		device = torch.device(cfg.device)
		self.device = device
		self.mse = nn.MSELoss()

	def forward(self, image_hat, image):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		# image_dim = image.size()

	    # [-1 1] to [0 1]
		image_hat = (image_hat + 1) / 2
		image = (image + 1) / 2

		mse = self.mse(image_hat, image)
		total_loss = mse
		psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))

		return total_loss, psnr

	@staticmethod
	def get_performance_metric():
		return "PSNR"


class IT_SSIM(torch.nn.Module):
	def __init__(self, cfg):
		super(IT_SSIM, self).__init__()
		device = torch.device(cfg.device)
		self.device = device

	def forward(self, image_hat, image):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat+1)/2
		image = (image+1)/2

		ssim = ssim_(image_hat, image, data_range=1, size_average=True)
		total_loss = 1 - ssim

		return total_loss, ssim.clone().detach().cpu()

	@staticmethod
	def get_performance_metric():
		return "SSIM"


class IT_MSSSIM(torch.nn.Module):
	def __init__(self, cfg):
		super(IT_MSSSIM, self).__init__()
		device = torch.device(cfg.device)
		self.device = device

	def forward(self, image_hat, image):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat+1)/2
		image = (image+1)/2

		msssim = ms_ssim_(image_hat, image, data_range=1, size_average=True)
		total_loss = 1-msssim

		return total_loss, msssim.clone().detach().cpu()

	def get_performance_metric(self):
		return "MS-SSIM"


class FAIT_MSE(torch.nn.Module):
	def __init__(self, cfg):
		super(FAIT_MSE, self).__init__()
		device = cfg.device
		self.device = device
		self.mse = nn.MSELoss()
		self.CA_ratio = cfg.ratio
		self.gamma = cfg.gamma

	def forward(self, image_hat, image,decision=None):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat + 1) / 2
		image = (image + 1) / 2

		mse = self.mse(image_hat, image)
		total_loss = mse
		if decision:
			mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
			mask_loss = mask_loss.to(self.device)
			total_loss += mask_loss
		psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))

		return mse, psnr

	def get_performance_metric(self):
		return "PSNR"


class Freq_MSE(torch.nn.Module):
	def __init__(self, cfg):
		super(Freq_MSE, self).__init__()
		device = cfg.device
		self.device = device
		self.mse = nn.MSELoss()
		self.CA_ratio = cfg.ratio
		self.beta = cfg.beta

	def forward(self, image_hat, image, coarse, f, f_inv):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat + 1) / 2
		image = (image + 1) / 2

		mse = self.mse(image_hat, image)
		total_loss = mse
		aux_loss1 = self.mse(f_inv(f(image)), image)
		aux_loss2 = self.mse(f(f_inv(coarse)), coarse.detach())
		total_loss = total_loss + self.beta * (aux_loss1 + aux_loss2)
		psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))

		return total_loss, psnr

	def get_performance_metric(self):
		return "PSNR"


class Gaussian_MSE(torch.nn.Module):
	def __init__(self, cfg):
		super(Gaussian_MSE, self).__init__()
		device = cfg.device
		self.device = device
		self.mse = nn.MSELoss()
		self.CA_ratio = cfg.ratio
		self.beta = cfg.beta

	def forward(self, image_hat, image, p_m, p_s, q_m, q_s):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat + 1) / 2
		image = (image + 1) / 2

		mse = self.mse(image_hat, image)
		total_loss = mse
		aux_loss = kl_divergence(p_m, p_s, q_m, q_s)
		total_loss = total_loss + self.beta * aux_loss
		psnr = 10 * (np.log(1. / mse.clone().detach().cpu()) / np.log(10))

		return total_loss, psnr

	def get_performance_metric(self):
		return "PSNR"


class FAIT_SSIM(torch.nn.Module):
	def __init__(self, cfg):
		super(FAIT_SSIM, self).__init__()
		device = cfg.device
		self.device = device
		self.mse = nn.MSELoss()
		self.CA_ratio = cfg.ratio
		self.gamma = cfg.gamma

	def forward(self, image_hat, image,decision=None):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat+1)/2
		image = (image+1)/2

		ssim = ssim_(image_hat, image, data_range=1, size_average=True)
		total_loss = 1-ssim
		if decision:
			mask_loss = 2*self.gamma*(torch.mean(torch.cat(decision,dim=1),dim=(0,1))-self.CA_ratio)**2
			mask_loss = mask_loss.to(self.device)
			total_loss += mask_loss

		return total_loss, ssim.clone().detach().cpu()

	def get_performance_metric(self):
		return "SSIM"

class imagewisePSNR(torch.nn.Module):
	def __init__(self, cfg):
		super(imagewisePSNR, self).__init__()
		device = cfg.device
		self.device = device
		self.mse = nn.MSELoss(reduction='none')

	def forward(self, image_hat, image):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat+1)/2
		image = (image+1)/2

		unreduced_mse = self.mse(image_hat, image)
		image_wise_mse = unreduced_mse.mean(dim=[i for i in range(1,len(image.size()))]).reshape(-1).clone().detach().cpu()

		image_wise_psnr = 10 * (np.log(1. / image_wise_mse) / np.log(10))
		return image_wise_psnr


class imagewiseSSIM(torch.nn.Module):
	def __init__(self, cfg):
		super(imagewiseSSIM, self).__init__()
		device = cfg.device
		self.device = device

	def forward(self, image_hat, image):
		# inputs => N x C x H x W
		image_hat = image_hat.to(self.device)
		image = image.to(self.device)
		#image_dim = image.size()

	  # [-1 1] to [0 1]
		image_hat = (image_hat+1)/2
		image = (image+1)/2

		unreduced_SSIM = ssim_(image_hat, image, data_range=1, size_average=False)
		image_wise_SSIM = unreduced_SSIM.reshape(-1).clone().detach().cpu()

		return image_wise_SSIM


def kl_divergence(prior_mu, prior_logvar, post_mu, post_logvar, reduction='mean'):
	# KL = 0.5 * sum( exp(logvar_q - logvar_p) + (mu_q - mu_p)^2 * exp(-logvar_p) - 1 + logvar_p - logvar_q )
	post_mu, post_logvar = post_mu.detach(), post_logvar.detach()
	var_ratio = torch.exp(post_logvar - prior_logvar)  # exp(logvar_q - logvar_p)
	mu_diff_sq = (post_mu - prior_mu).pow(2)  # (mu_q - mu_p)^2
	precision_p = torch.exp(-prior_logvar)  # 1/var_p = exp(-logvar_p)

	kl_element = 0.5 * (var_ratio + mu_diff_sq * precision_p - 1 + prior_logvar - post_logvar)
	# kl_element 形状: (batch_size, latent_dim)

	# 对 latent_dim 求和，得到每个样本的 KL 散度
	kl_per_sample = kl_element.sum(dim=1)  # 形状 (batch_size,)

	if reduction == 'mean':
		return kl_per_sample.mean()
	elif reduction == 'sum':
		return kl_per_sample.sum()
	else:
		return kl_per_sample  # 返回每个样本的 KL 散度




