# EgoFound3R: End-to-End Egocentric Hand Reconstruction in World Space with Point-Wise Interaction Attributes

[**Hongming Fu**](https://frank-f2022.github.io/)<sup>1,2</sup>, **Jingcheng Shi**<sup>2,3</sup>, [**Wenjia Wang**](https://wenjiawang0312.github.io/)<sup>4</sup>, **Binhua Zuo**<sup>2,\*</sup>, [**Bo Zhao**](https://www.bozhao.me/)<sup>1,\*</sup>

<sup>1</sup> School of AI, Shanghai Jiao Tong University, <sup>2</sup> QuicRobot, <sup>3</sup> Rutgers University, <sup>4</sup> The University of Hong Kong

<sup>*</sup> Corresponding authors

<a href="https://mint-sjtu.github.io/EgoFound3R.io/"><img src="https://img.shields.io/badge/Project_Page-EgoFound3R-1f6feb?style=flat-square" height="22" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2610.01210"><img src="https://img.shields.io/badge/arXiv-2610.01210-b31b1b?style=flat-square" height="22" alt="arXiv"></a>

Inference code for EgoFound3R: metric 3D hand reconstruction with interaction
attributes (visibility, contact, distance) in world space from egocentric video,
and 2D visualization overlays.

## Installation

```bash
pip install -r requirements.txt
```

VGGT-Omega is vendored under `other/vggt-omega`; no separate install is needed.

## Checkpoints

1. **EgoFound3R**: download `EgoFound3R.pt` from
   [huggingface.co/Frank-F2021/EgoFound3R](https://huggingface.co/Frank-F2021/EgoFound3R)
   and pass it with `--checkpoint`.
2. **VGGT-Omega**: the frozen geometry backbone. Download `vggt_omega_1b_512.pt`
   from the [model page](https://huggingface.co/facebook/VGGT-Omega) and point
   `vggt_checkpoint_path` in `configs/default.toml` at it.

## Usage

```bash
python infer_marker_video.py \
    --video-path path/to/video.mp4 \
    --checkpoint path/to/egofound3r.pt \
    --output-dir outputs/video
```

The script writes:

| File | Contents |
| --- | --- |
| `inference_output.pt` | all predicted tensors (camera, depth, intrinsics, hand root / joints / vertices, visibility, contact) |
| `summary.json` | scalars, provenance and timings |
| `overlay.mp4` | 2D visualization of the predicted hand geometry |
| `overlay_multimode.mp4` | 2D visualization of geometry, visibility, contact and distance, 2x2 |

Useful flags:

| Flag | Effect |
| --- | --- |
| `--device auto\|cpu\|cuda\|cuda:N` | compute device (default `auto`) |
| `--no-save-overlay` | skip both overlay videos |
| `--no-overlay-frame-images` | write only the videos, not per-frame PNGs |
| `--upsample-mano` | add independent 778-vertex meter outputs |
| `--overlay-marker-count 195\|778` | marker count used for the overlays |
| `--max-frames N` | process only the first N frames |
| `--chunk-seconds / --overlap-seconds` | sliding-window length and overlap |

## License

Copyright (c) 2026 MINT-SJTU. Our code and model weights are released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](./LICENSE)
(CC BY-NC-SA 4.0): academic research and educational use only, commercial use is
not permitted. Third-party components keep their own terms: [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) and [microsoft/MeshGraphormer](https://github.com/microsoft/MeshGraphormer).

## Citation

```bibtex
@misc{fu2026egofound3rendtoendegocentrichand,
  title={EgoFound3R: End-to-End Egocentric Hand Reconstruction in World Space with Point-Wise Interaction Attributes},
  author={Hongming Fu and Jingcheng Shi and Wenjia Wang and Binhua Zuo and Bo Zhao},
  year={2026},
  eprint={2610.01210},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.01210}
}
```
