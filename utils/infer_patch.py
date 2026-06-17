import torch
import numpy as np
import torch.nn.functional as F
import rasterio
from .misc import *
import os
import tifffile

""" normalisation du glo30 """
# mean et std pour normaliser le glo30
GLOBAL_MEAN = 696.0574
GLOBAL_STD = 875.0319


def extract_patches(x, h, w, sh, sw):
    # patch de l'image en entrée
    B, C, H, W = x.shape

    pad_h = (sh - (H - h) % sh) % sh
    pad_w = (sw - (W - w) % sw) % sw

    x = F.pad(x, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)

    H_pad, W_pad = x.shape[-2:]

    patches = F.unfold(
        x,
        kernel_size=(h, w),
        stride=(sh, sw)
    )

    patches = patches.transpose(1, 2)
    patches = patches.view(B, -1, C, h, w)

    return patches, H_pad, W_pad

def reconstruct_from_patches_weighted_sigma(patches, sigmas, H, W, H_pad, W_pad, hpatch, wpatch, sh, sw):
    # réassemble les patch en prenant en priorité les variances les plus faibles
    B, N, C, h, w = patches.shape

    patches = patches.view(B, N, C * h * w)
    patches = patches.transpose(1, 2)

    sigmas = sigmas.view(B, N, C * h * w)
    sigmas = sigmas.transpose(1, 2)

    # sigma weights
    sigmas_rev =  1/(sigmas.clamp(min=0.1)**2)
    sigma_sum = F.fold(
        sigmas_rev,
        output_size=(H_pad, W_pad),
        kernel_size=(h, w),
        stride=(sh, sw)
    )
    # repatch
    sigma_sum_patched, _, _ = extract_patches( sigma_sum, hpatch, wpatch, sh, sw )
    sigma_sum_patched = sigma_sum_patched.view(B, N, C * h * w)
    sigma_sum_patched = sigma_sum_patched.transpose(1, 2)

    weights = sigmas_rev / sigma_sum_patched
    
    out = F.fold(
        patches * weights,
        output_size=(H_pad, W_pad),
        kernel_size=(h, w),
        stride=(sh, sw)
    )

    out_sigma = F.fold(
        sigmas * weights,
        output_size=(H_pad, W_pad),
        kernel_size=(h, w),
        stride=(sh, sw)
    )

    return out[:, :, :H, :W], out_sigma[:, :, :H, :W] 


def open_tif( p, to_tensor = False, resize = False):
    with rasterio.open(p) as src:
        data = src.read().astype(np.float32).squeeze()
        # print(src.crs)
    if to_tensor:
        return to_tensor_rs(data, resize)
    return data

def to_tensor_rs(  array, resize = False ):
    n_unsqueeze = 2 if array.ndim == 2 else 1
    ts = torch.from_numpy( array )
    for _ in range(n_unsqueeze):
        ts = ts.unsqueeze(0)
    # target_shape = (512,512) if ts.shape
    # print(ts.shape)
    if resize:
        return F.interpolate( ts, (512,512))
    return ts
    
def get_input( all_paths, to_tensor = True, resize = False, band_offset = -1000, band_quantification=10000, only_rgb = False ):

    b02 = open_tif( all_paths['b02_path'] )
    b03 = open_tif( all_paths['b03_path'] )
    b04 = open_tif( all_paths['b04_path'] )
    
    if only_rgb: return np.stack( (b02, b03, b04) ) / band_quantification
    b8a = open_tif( all_paths['b8a_path'] )
    b09 = open_tif( all_paths['b09_path'] )
    b11 = open_tif( all_paths['b11_path'] )
    glo30 = open_tif( all_paths['glo30_path'] )

    m = b02 != 0

    b02[m] = (b02[m] + band_offset) / band_quantification
    b03[m] = (b03[m] + band_offset) / band_quantification
    b04[m] = (b04[m] + band_offset) / band_quantification
    b8a[m] = (b8a[m] + band_offset) / band_quantification
    b09[m] = (b09[m] + band_offset) / band_quantification
    b11[m] = (b11[m] + band_offset) / band_quantification

    bands = np.stack( (b02, b03, b04, b8a, b09, b11, glo30 ) )

    if to_tensor:
        bands = torch.from_numpy( bands ).unsqueeze(0)
        if resize: bands = F.interpolate( bands, (512,512))
    return bands
    
def filter_list_by_index( list_, idx_ ):
    return [ list_[v] for v in idx_ ]


def infer_dataset( dataset, name, model_ConvNeXtDPT, temporal_head, infer_size = 512, step_size = 128, use_time_forward = True, max_bs = 8, p_out = None, encoder_type = 'gla3',
                    device = 'cuda', model_type='multi', progress_callback=None ):

    sh, sw = infer_size - step_size, infer_size - step_size
    d = dataset[ name ]

    time_forward = max_bs // 2
    B = len(d['acq_date'])
    idxs = np.arange( 0, B )

    # --- check which dates already have all outputs saved ---
    def all_outputs_exist(acq_date):
        if p_out is None:
            return False
        return all(os.path.exists( os.path.join(p_out, f'{prefix}_{name}_' + acq_date + '.tif'))
                   for prefix in ('pred', 'sigma', 'refined'))

    cached = [all_outputs_exist(acq_date) for acq_date in d['acq_date']]
    missing_idxs = [i for i, c in enumerate(cached) if not c]

    if use_time_forward:
        if B <= time_forward:
            iter_ = range( 0, B, max_bs )
            do_sub = False
        else:
            iter_ = range( time_forward, B, time_forward )
            do_sub = True
    else:
        iter_ = range( 0, B, max_bs )
        do_sub = False
    
    full_pred_depth = None
    full_pred_var   = None
    full_pred_count = None

    for i in iter_:
        
        if progress_callback is not None:
            progress_callback(i)
            
        if do_sub:
            sup = idxs[i:i+time_forward]
            down_idx = i//time_forward
            down = idxs[::down_idx]
            sub_idx = np.concatenate( (down[:time_forward], sup))
        elif use_time_forward:
            sub_idx = idxs
        else:
            sub_idx = idxs[i:i+max_bs]
        
        batch_missing = [j for j in sub_idx if j in missing_idxs]
        if len(batch_missing) == 0:
            continue

        sub_acq_date   = filter_list_by_index( d['acq_date'],    sub_idx )
        paths_bands    = filter_list_by_index( d['band_path'],   sub_idx )
        sub_safe_folder = filter_list_by_index( d['safe_folder'], sub_idx )

        data_bands = [(p, get_input(p)) for p in paths_bands]

        max_shape = max(x.shape for _, x in data_bands)
        bands_and_glo = [x for _, x in data_bands if x.shape == max_shape]
        bands_and_glo = torch.vstack( bands_and_glo )

        B_cur, C, H, W = bands_and_glo.shape
        # print( 'bands_and_glo.shape', bands_and_glo.shape )
        if full_pred_depth is None:
            full_pred_depth = torch.zeros( len(d['acq_date']), 1, H, W )
            full_pred_var   = torch.zeros( len(d['acq_date']), 1, H, W )
            full_pred_count = torch.zeros( len(d['acq_date']), 1, H, W )

        sub_acq_date = [ s for s, (_, x) in zip(sub_acq_date, data_bands) if x.shape == max_shape ]
        solstice_delay = days_from_winter_solstice( [ datetime.strptime(t,'%Y%m%d') for t in sub_acq_date])
        time_delay     = timestamp_to_delay(        [ datetime.strptime(t,'%Y%m%d') for t in sub_acq_date], encoder_type)

        bands_and_glo[:, -1] = (bands_and_glo[:, -1] - GLOBAL_MEAN) / GLOBAL_STD

        if (H > infer_size) or (W > infer_size):
            patches, H_pad, W_pad = extract_patches( bands_and_glo, infer_size, infer_size, sh, sw )
            is_patched = True
        else:
            patches = bands_and_glo[:, None]
            H_pad, W_pad = H, W
            is_patched = False

        T, N, C, hp, wp = patches.shape
        pred_patch = torch.zeros( T, N, 2, hp, wp )
        
        with torch.no_grad():
            for n in range(N):
                cur_input = patches[:, n]

                timedelta = torch.zeros_like( cur_input[:, [0]] ).to(device)

                if model_type == 'multi':
                    feats = model_ConvNeXtDPT.get_convnext_features( cur_input.float().to(device) )
                    preds, _ = temporal_head(feats, cur_input.shape[0], position=[timedelta, solstice_delay, time_delay.to(device)])
                    del feats
                else:
                    preds = model_ConvNeXtDPT( cur_input.float().to(device) )
                pred_patch[:, n] += preds.cpu()
                del preds, cur_input

                torch.cuda.empty_cache()
                
        # continue
        pred_depth = pred_patch[:, :, [0]]
        pred_var   = pred_patch[:, :, [1]].exp()
        del pred_patch

        if is_patched:
            pred_depth, pred_var = reconstruct_from_patches_weighted_sigma(
                pred_depth, pred_var, H, W, H_pad, W_pad, infer_size, infer_size, sh, sw)
        else:
            pred_depth = pred_depth.squeeze(1)
            pred_var   = pred_var.squeeze(1)

        full_pred_depth[ sub_idx ] += pred_depth
        full_pred_var[   sub_idx ] += pred_var
        full_pred_count[ sub_idx ] += 1

        del pred_depth, pred_var
    # return None,None,None
    # --- assemble final outputs, loading from disk where cached ---
    sample_date = d['acq_date'][0]
    if full_pred_depth is not None:
        _, _, H, W = full_pred_depth.shape
    else:
        # all dates were cached; read shape from first cached pred
        tmp = tifffile.imread( os.path.join( p_out, f'pred_{name}_' + sample_date + '.tif'))
        H, W = tmp.shape[-2], tmp.shape[-1]
        del tmp

    final_pred_depth = torch.zeros( len(d['acq_date']), 1, H, W )
    final_pred_var   = torch.zeros( len(d['acq_date']), 1, H, W )

    for j, acq_date in enumerate(d['acq_date']):
        if cached[j]:
            final_pred_depth[j, 0] = torch.from_numpy(
                tifffile.imread( os.path.join(p_out, f'pred_{name}_'  + acq_date + '.tif')))
            final_pred_var[j, 0]   = torch.from_numpy(
                tifffile.imread( os.path.join( p_out, f'sigma_{name}_' + acq_date + '.tif')))
        else:
            count = full_pred_count[j]
            final_pred_depth[j] = full_pred_depth[j] / count
            final_pred_var[j]   = full_pred_var[j]   / count

    glo30_ = get_input( d['band_path'][0] )[0, -1]
    refined = glo30_ - final_pred_depth.squeeze()

    if p_out is not None:
        for j, acq_date in enumerate(d['acq_date']):
            if not cached[j]:   # don't rewrite files we just loaded
                tifffile.imwrite( os.path.join(p_out, f'pred_{name}_'    + acq_date + '.tif'), final_pred_depth[j].squeeze().numpy())
                tifffile.imwrite(os.path.join(p_out, f'sigma_{name}_'   + acq_date + '.tif'), final_pred_var[j].squeeze().numpy())
                tifffile.imwrite( os.path.join(p_out, f'refined_{name}_' + acq_date + '.tif'), refined[j].squeeze().numpy())
    torch.cuda.empty_cache()
    return refined, final_pred_depth, final_pred_var
