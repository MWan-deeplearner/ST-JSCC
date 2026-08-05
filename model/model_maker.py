from model import JSCC

def get_model_info(cfg):
    model_info = dict()
    model_info['chan_type'] = cfg.chan_type
    model_info['color_channel'] = cfg.data_info.color_channel
    model_info['rcpp'] = cfg.rcpp #reverse of channel per pixel
    model_info['window_size'] = 8
    model_info['ratio'] = 0.5
    cfg.ratio = model_info['ratio'] #importance ratio
    model_info['gamma'] = 0.5
    cfg.gamma = model_info['gamma']
    model_info['ratio1'] = 0.5
    cfg.ratio1 = model_info['ratio1'] # encoder's importance ratio
    model_info['ratio2'] = 0.5
    cfg.ratio2 = model_info['ratio2'] # decoder's importance ratio
    model_info['swap'] = 0.0
    model_info['window_size_list'] = [8,8,8,8] #[8,8,8,8]
    model_info['num_heads_list'] = [4, 6, 8, 10] ##careful! n_feats_list[i]/num_heads_list[i] should be integer
    model_info['input_resolution'] = cfg.input_resolution
    model_info['n_block_list'] = [2,2,2,2]
    if cfg.model_name in ["ConvJSCC", "ResJSCC"]:
        model_info['n_feats_list'] = [64, 64, 64, 64]
    elif cfg.model_name in ["SwinJSCC", "LAJSCC"]:
        model_info['n_feats_list'] = [40, 60, 80, 160]
    elif cfg.model_name == "LICRFJSCC":
        model_info['n_feats_list'] = [64, 96, 128, 180]
    elif cfg.model_name == "FAJSCC":
        model_info['n_feats_list'] = [40, 60, 80, 260]
    else:
        model_info = cfg
    return model_info


def ModelMaker(cfg):
    model_info = get_model_info(cfg)
    model = getattr(JSCC, cfg.model_name)(model_info)
    return model
    
    
 