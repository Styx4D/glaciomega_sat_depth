<div align="center">

# GlacioMega
# Surveying surface elevation change from Sentinel-2 monocular optical satellite time series

**Alexandre Baratier** (Styx4D), **Pierre Lemaire** (Styx4D), **Dawa Derksen** (CNES), **Benoit Urruty** (Styx4D), **Johan Berthet** (Styx4D)

> *Styx4D, Le Bourget-du-Lac, France — CNES, Toulouse, France*

[[Paper]](#) · [[Data]](https://zenodo.org/records/20491723) · [[Models]](#pretrained-models)

![GlacioMega overview](fig/figure_GA.svg)

</div>

## Overview

GlacioMega is a multi-temporal deep learning framework that estimates topographic change and refines the Copernicus GLO-30 digital surface model from Sentinel-2 optical time series — no stereo imagery, no ground data required at inference time.

Trained on nearly **100,000 Sentinel-2/LiDAR pairs** across four European countries, GlacioMega reduces GLO-30 RMSE by **39%** and enables large-scale monitoring of elevation dynamics.

---

## Pretrained Models

Three checkpoints are available, covering monotemporal and multi-temporal use cases:

| Model | Description | Download |
|---|---|---|
| **ConvNextDPT** | Monotemporal backbone. Use standalone or as encoder for multi-temporal variants. | [link](https://zenodo.org/records/22125346/files/convNext_base_DPT_finetune.pth?download=1) |
| **Temporal DPT (TPE)** | Multi-temporal model. Best validation RMSE on the benchmark dataset. | [link](https://zenodo.org/records/22125346/files/convNext_base_DPT_multitemp_SimpleRelativePos.pth?download=1) |
| **Temporal DPT (MTPE)** | Multi-temporal model. Best accuracy on the GlacioClim glacier validation set. | [link](https://zenodo.org/records/22125346/files/convNext_base_DPT_multitemp_MultiRelativePos.pth?download=1) |

---

## Data

Qualitative analysis data and the GlacioClim validation set used in the paper are available **[here](https://zenodo.org/records/20491723)**

---

## Usage

### Installation

```bash
git clone https://github.com/Styx4D/glaciomega_sat_depth.git
cd glaciomega_sat_depth
pip install -r requirements.txt
```

### Running inference

Three notebooks are provided for different use cases:

| Notebook | Purpose |
|---|---|
| `inference_test.ipynb` | Run inference on the qualitative paper dataset |
| `inference_glacioclim.ipynb` | Evaluate on the GlacioClim glacier dataset |
| `inference_custom_zone.ipynb` | Run inference on your own Sentinel-2 time series |

---

## License

Code and models are released under the **CC BY-NC 4.0** license.
For commercial use, please [contact us](mailto:styx@styx.earth).

---

## Acknowledgments
This work was supported by the Centre National d’Etudes Spatiales (CNES)  through an R\&T mechanism on a 'data hybridization' challenge.

---

## Citation

If you use GlacioMega in your research, please cite:

```bibtex
@article{baratier2025glaciomega,
  title     = {Surveying surface elevation change from {Sentinel-2} monocular optical satellite time series},
  author    = {Baratier, Alexandre and Lemaire, Pierre and Derksen, Dawa and Urruty, Benoit and Berthet, Johan},
  journal   = {Remote Sensing of Environment},
  year      = {2025},
}
```
