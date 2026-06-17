import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import ipywidgets as widgets
from IPython.display import display
from .infer_patch import get_input, infer_dataset

def hillshade(array, azimuth=315, angle_altitude=45):
    """
    Calcule un ombrage simulé (hillshade) à partir d'un modèle numérique de surface.

    Paramètres
    ----------
    array : np.ndarray (H, W), Modèle numérique d'élévation (MNS ou MNT).
    azimuth : float, optionnel
    angle_altitude : float, optionnel
    """
    azimuth = 360.0 - azimuth

    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)

    azm_rad = azimuth * np.pi / 180.
    alt_rad = angle_altitude * np.pi / 180.

    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)

    return 255 * (shaded + 1) / 2

def enhance_img(image):
    """
    Améliore la dynamique d'une image par fusion multi-exposition simulée.
    Paramètres
    ----------
    image : np.ndarray, images sentinelles à valeur brut (0 à ~ 10 000).
    """
    scales = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
    scales = [s * 10000 for s in scales]
    enhanced_images = []
    for s in scales:
        img_scale = image.copy() / s
        img_scale[img_scale > 1] = 1
        enhanced_images.append(img_scale)
    enhanced_image = np.sum(enhanced_images, axis=0)
    return enhanced_image / enhanced_image.max()


def clahe_hsv( img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
 
    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit = 15.0, tileGridSize = (10,10))
    v = clahe.apply(v)
 
    hsv_img = np.dstack((h,s,v))
 
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    
    return rgb
 
   
def rgb_glaciomega(img_rgb):
    """
    Pipeline de rendu visuel adapté aux images de surfaces enneigées.
    Fusionne un HDR simulée et CLAHE
    Paramètres
    ----------
    img_rgb : np.ndarray (H, W, 3), Image RGB d'entrée à valeurs réelles.
    """
    img_hdr = enhance_img(img_rgb)
    img_min = img_rgb[img_rgb != 0].min()
    img_max = img_rgb.max()
    img_norm = (img_rgb - img_min) / (img_max - img_min)
    img_clahe = clahe_hsv((img_norm * 255).astype(np.uint8)) / 255.0
    return img_hdr * 0.8 + img_clahe * 0.2

""" interactive plot """
def interactive_plot(dataset, refined, full_pred_depth, full_pred_var ):
    # plot interactife dans lequel on donne une donnée déjà inférencé

    # ── state ─────────────────────────────────────────────────────────────────
    state = dict(
        data_name   = list(dataset.keys())[0],
        current_idx = 0,
        PLOT_MAX    = 20,
        ILLUM_RATIO = 1.,
        t0_refined  = None,
        t0_rgb      = None,
        t0_date     = None,
        fig         = None,
        axes        = None,
    )

    # ── widgets ───────────────────────────────────────────────────────────────
    dataname_sel = widgets.Dropdown(description="data:", options=list(dataset.keys()))
    vmax_slider  = widgets.FloatSlider(description='vmax', min=0, max=100, value=20, continuous_update=True)
    illum_slider = widgets.FloatSlider(description='illum', min=0, max=2,  value=1,  continuous_update=True)
    idx_slider   = widgets.IntSlider(
        description="Index:",
        min=0,
        max=len(dataset[state['data_name']]['band_path']) - 1,
        value=0,
        continuous_update=False,
    )
    btn_prev   = widgets.Button(description="⏮ Prev")
    btn_next   = widgets.Button(description="Next ⏭")
    btn_set_t0 = widgets.Button(description="Set current as t0")
    out        = widgets.Output()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _init_t0(dname):
        p0    = dataset[dname]['band_path'][0]
        date0 = dataset[dname]['acq_date'][0]
        img0  = get_input(p0, to_tensor=False, only_rgb=True)
        state['t0_refined'] = refined[0].squeeze().numpy()
        state['t0_rgb']     = rgb_glaciomega(img0[:3].transpose(1, 2, 0).clip(0, 1))
        state['t0_date']    = date0

    def _build_figure():
        fig, axes_grid = plt.subplots(2, 3, figsize=(15, 11), sharex=True, sharey=True)
        state['fig']  = fig
        state['axes'] = axes_grid.flatten()
        for ax in state['axes']:
            ax.axis("off")
        plt.tight_layout()
        fig.canvas.header_visible = False
        with out:
            out.clear_output(wait=True)
            display(fig.canvas)

    def update_plot(idx):
        if state['fig'] is None:
            _build_figure()

        dname   = state['data_name']
        date    = dataset[dname]['acq_date'][idx]
        p       = dataset[dname]['band_path'][idx]
        img     = get_input(p, to_tensor=False, only_rgb=True)
        rgb     = rgb_glaciomega(img[:3].transpose(1, 2, 0).clip(0, 1))
        ref_np  = refined[idx].squeeze().numpy()
        depth_np = full_pred_depth[idx].squeeze().numpy()
        var_np   = full_pred_var[idx].squeeze().numpy()

        try:
            diff = ref_np - state['t0_refined']
        except Exception:
            diff = np.zeros_like(ref_np)

        PLOT_MAX    = state['PLOT_MAX']
        ILLUM_RATIO = state['ILLUM_RATIO']
        ax_t0, ax_rgb, ax_diff, ax_ref, ax_depth, ax_var = state['axes']

        panels = [
            (ax_t0,    (state['t0_rgb'][:,:,::-1] * ILLUM_RATIO).clip(0,1), dict(vmax=0.8)),
            (ax_rgb,   (rgb[:,:,::-1] * ILLUM_RATIO).clip(0,1),              dict(vmax=0.8)),
            (ax_diff,  diff,                           dict(vmin=-PLOT_MAX, vmax=PLOT_MAX, cmap="seismic")),
            (ax_ref,   hillshade(state['t0_refined']),              dict(cmap="gray")),
            (ax_depth,   hillshade(ref_np),              dict(cmap="gray")),
            (ax_var,   var_np,                         dict(vmin=0, vmax=PLOT_MAX, cmap="magma")),
        ]

        for ax, data, kw in panels:
            for im in ax.images:
                im.remove()
            ax.imshow(data, **kw)

        ax_t0.set_title(f"RGB  t0 = {state['t0_date']}")
        ax_rgb.set_title(f"RGB  t = {date}")
        ax_diff.set_title("Δ Refined  (t − t0)")
        ax_ref.set_title("Shaded t0")
        ax_depth.set_title("Shaded t")
        ax_var.set_title("Variance")
        # state['fig'].suptitle(f"{dname}  —  {date}")

        for ax in state['axes']:
            ax.axis("off")

        state['fig'].canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def on_prev(_):    idx_slider.value = max(idx_slider.min, idx_slider.value - 1)
    def on_next(_):    idx_slider.value = min(idx_slider.max, idx_slider.value + 1)

    def on_set_t0(_):
        idx  = idx_slider.value
        dname = state['data_name']
        date = dataset[dname]['acq_date'][idx]
        p    = dataset[dname]['band_path'][idx]
        img  = get_input(p, to_tensor=False, only_rgb=True)
        state['t0_refined'] = refined[idx].squeeze().numpy()
        state['t0_rgb']     = rgb_glaciomega(img[:3].transpose(1, 2, 0).clip(0, 1))
        state['t0_date']    = date
        update_plot(idx)

    def on_data_change(change):
        state['data_name'] = change["new"]
        idx_slider.unobserve(on_idx_change, names="value")
        idx_slider.max   = len(dataset[state['data_name']]['band_path']) - 1
        idx_slider.value = 0
        idx_slider.observe(on_idx_change, names="value")
        _init_t0(state['data_name'])
        update_plot(0)

    def on_idx_change(change):
        state['current_idx'] = change["new"]
        update_plot(change["new"])

    def on_vmax_change(change):
        state['PLOT_MAX'] = change["new"]
        update_plot(state['current_idx'])

    def on_illum_change(change):
        state['ILLUM_RATIO'] = change["new"]
        update_plot(state['current_idx'])

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_set_t0.on_click(on_set_t0)
    dataname_sel.observe(on_data_change, names="value")
    idx_slider.observe(on_idx_change,    names="value")
    vmax_slider.observe(on_vmax_change,  names="value")
    illum_slider.observe(on_illum_change, names="value")

    # ── launch ─────────────────────────────────────────────────────────────────
    _init_t0(state['data_name'])

    controls = widgets.HBox([btn_prev, btn_next, btn_set_t0])
    ui       = widgets.VBox([idx_slider, controls, dataname_sel, widgets.HBox([vmax_slider, illum_slider])])
    display(ui, out)
    update_plot(0)

def interactive_plot_with_pred(dataset, model_ConvNeXtDPT, temporal_head, infer_size=512, step_size=128,
                               use_time_forward=True, max_bs=8, p_out=None, encoder_type='gla3',
                               device='cuda', model_type='multi', show_title=False):

    pred_cache = {}

    state = dict(
        data_name   = list(dataset.keys())[0],
        current_idx = 0,
        PLOT_MAX    = 20,
        ILLUM_RATIO = 1.,
        t0_refined  = None,
        t0_rgb      = None,
        t0_date     = None,
        fig         = None,
        axes        = None,
    )

    def _get_preds(dname):
        if dname not in pred_cache:
            n_images = len(dataset[dname]['band_path'])

            # ── progress bar ──────────────────────────────────────────────────
            progress_bar = widgets.IntProgress(
                value=0, min=0, max=n_images,
                description='Inférence:',
                bar_style='info',        # 'success', 'info', 'warning', 'danger'
                style={'description_width': 'initial', 'bar_color': '#4A90D9'},
                layout=widgets.Layout(width='80%'),
            )
            progress_label = widgets.Label(value=f"0 / {n_images} images")
            progress_box   = widgets.VBox([progress_bar, progress_label])

            with out:
                out.clear_output(wait=True)
                display(progress_box)

            def progress_callback(i):
                progress_bar.value   = i + 1
                progress_label.value = f"{i + 1} / {n_images} images"

            # ── inference ─────────────────────────────────────────────────────
            refined, full_pred_depth, full_pred_var = infer_dataset(
                dataset, dname,
                model_ConvNeXtDPT=model_ConvNeXtDPT,
                temporal_head=temporal_head,
                infer_size=infer_size,
                step_size=step_size,
                use_time_forward=use_time_forward,
                max_bs=max_bs,
                p_out=p_out,
                encoder_type=encoder_type,
                device=device,
                model_type=model_type,
                progress_callback=progress_callback,  
            )

            progress_bar.bar_style = 'success'
            progress_label.value   = f"✓ {n_images} / {n_images} — terminé"

            pred_cache[dname] = (refined, full_pred_depth, full_pred_var)

        return pred_cache[dname]

    # ── widgets ───────────────────────────────────────────────────────────────
    dataname_sel = widgets.Dropdown(description="data:", options=list(dataset.keys()))
    vmax_slider  = widgets.FloatSlider(description='vmax', min=0, max=100, value=20, continuous_update=True)
    illum_slider = widgets.FloatSlider(description='illum', min=0, max=2,  value=1,  continuous_update=True)
    idx_slider   = widgets.IntSlider(
        description="Index:",
        min=0,
        max=len(dataset[state['data_name']]['band_path']) - 1,
        value=0,
        continuous_update=False,
    )
    btn_prev   = widgets.Button(description="⏮ Prev")
    btn_next   = widgets.Button(description="Next ⏭")
    btn_set_t0 = widgets.Button(description="Set current as t0")
    out        = widgets.Output()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _init_t0(dname):
        refined, _, _ = _get_preds(dname)
        p0    = dataset[dname]['band_path'][0]
        date0 = dataset[dname]['acq_date'][0]
        img0  = get_input(p0, to_tensor=False, only_rgb=True)
        state['t0_refined'] = refined[0].squeeze().numpy()
        state['t0_rgb']     = rgb_glaciomega(img0[:3].transpose(1, 2, 0).clip(0, 1))
        state['t0_date']    = date0

    def _build_figure():
        fig, axes_grid = plt.subplots(2, 3, figsize=(15, 11), sharex=True, sharey=True)
        state['fig']  = fig
        state['axes'] = axes_grid.flatten()
        for ax in state['axes']:
            ax.axis("off")
        plt.tight_layout()
        fig.canvas.header_visible = False
        with out:
            out.clear_output(wait=True)
            display(fig.canvas)

    def update_plot(idx):
        if state['fig'] is None:
            _build_figure()

        dname            = state['data_name']
        refined, full_pred_depth, full_pred_var = _get_preds(dname)
        date             = dataset[dname]['acq_date'][idx]
        p                = dataset[dname]['band_path'][idx]
        img              = get_input(p, to_tensor=False, only_rgb=True)
        rgb              = rgb_glaciomega(img[:3].transpose(1, 2, 0).clip(0, 1))
        ref_np           = refined[idx].squeeze().numpy()
        var_np           = full_pred_var[idx].squeeze().numpy()

        try:
            diff = ref_np - state['t0_refined']
        except Exception:
            diff = np.zeros_like(ref_np)

        PLOT_MAX    = state['PLOT_MAX']
        ILLUM_RATIO = state['ILLUM_RATIO']
        ax_t0, ax_rgb, ax_diff, ax_ref, ax_depth, ax_var = state['axes']

        panels = [
            (ax_t0,    (state['t0_rgb'][:,:,::-1] * ILLUM_RATIO).clip(0,1), dict(vmax=0.8)),
            (ax_rgb,   (rgb[:,:,::-1] * ILLUM_RATIO).clip(0,1),              dict(vmax=0.8)),
            (ax_diff,  diff,                           dict(vmin=-PLOT_MAX, vmax=PLOT_MAX, cmap="seismic")),
            (ax_ref,   hillshade(state['t0_refined']),              dict(cmap="gray")),
            (ax_depth,   hillshade(ref_np),              dict(cmap="gray")),
            (ax_var,   var_np,                         dict(vmin=0, vmax=PLOT_MAX, cmap="magma")),
        ]

        for ax, data, kw in panels:
            for im in ax.images:
                im.remove()
            ax.imshow(data, **kw)


        ax_t0.set_title(f"RGB  t0 = {state['t0_date']}")
        ax_rgb.set_title(f"RGB  t = {date}")
        ax_diff.set_title("Δ Refined  (t − t0)")
        ax_ref.set_title("Shaded t0")
        ax_depth.set_title("Shaded t")
        ax_var.set_title("Variance")
        # state['fig'].suptitle(f"{dname}  —  {date}")

        for ax in state['axes']:
            ax.axis("off")

        state['fig'].canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────
    def on_prev(_):    idx_slider.value = max(idx_slider.min, idx_slider.value - 1)
    def on_next(_):    idx_slider.value = min(idx_slider.max, idx_slider.value + 1)

    def on_set_t0(_):
        idx   = idx_slider.value
        dname = state['data_name']
        refined, _, _ = _get_preds(dname)
        date  = dataset[dname]['acq_date'][idx]
        p     = dataset[dname]['band_path'][idx]
        img   = get_input(p, to_tensor=False, only_rgb=True)
        state['t0_refined'] = refined[idx].squeeze().numpy()
        state['t0_rgb']     = rgb_glaciomega(img[:3].transpose(1, 2, 0).clip(0, 1))
        state['t0_date']    = date
        update_plot(idx)

    def on_data_change(change):
        state['data_name'] = change["new"]
        idx_slider.unobserve(on_idx_change, names="value")
        idx_slider.max   = len(dataset[state['data_name']]['band_path']) - 1
        idx_slider.value = 0
        idx_slider.observe(on_idx_change, names="value")
        _init_t0(state['data_name'])
        update_plot(0)

    def on_idx_change(change):
        state['current_idx'] = change["new"]
        update_plot(change["new"])

    def on_vmax_change(change):
        state['PLOT_MAX'] = change["new"]
        update_plot(state['current_idx'])

    def on_illum_change(change):
        state['ILLUM_RATIO'] = change["new"]
        update_plot(state['current_idx'])

    btn_prev.on_click(on_prev)
    btn_next.on_click(on_next)
    btn_set_t0.on_click(on_set_t0)
    dataname_sel.observe(on_data_change, names="value")
    idx_slider.observe(on_idx_change,    names="value")
    vmax_slider.observe(on_vmax_change,  names="value")
    illum_slider.observe(on_illum_change, names="value")

    # ── launch ────────────────────────────────────────────────────────────────
    _init_t0(state['data_name'])
    controls = widgets.HBox([btn_prev, btn_next, btn_set_t0])
    ui       = widgets.VBox([idx_slider, controls, dataname_sel, widgets.HBox([vmax_slider, illum_slider])])
    display(ui, out)
    update_plot(0)