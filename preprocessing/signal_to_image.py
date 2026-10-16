# signal_to_image.py
# converts raw radar CSV files into spectrogram images that the CNN can train on
# basically the whole idea is: radar signal -> 2D image -> feed to CNN like any image classifier

import json
import logging
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image

import config

logger = logging.getLogger(__name__)


def read_raw_csv_signal(csv_path):
    """Read a radar CSV file and return it as a 2D numpy array."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    try:
        df = pd.read_csv(path, header=None).dropna(how="all", axis=1)
        matrix = df.to_numpy(dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError(f"bad matrix shape: {matrix.shape}")
        return matrix
    except Exception as exc:
        # sometimes pandas fails on weird csv formats, try numpy as fallback
        logger.warning(f"pandas failed on {path}: {exc}, retrying with genfromtxt")
        matrix = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0)
        return matrix


def convert_signal_to_spectrogram_image(signal_matrix, target_shape=config.SPECTROGRAM_SHAPE,
                                         colormap_name=config.COLORMAP_NAME):
    """
    Turn a raw 2D radar signal into a 128x128 RGB image.

    Steps i figured out after a lot of trial and error:
    1. log scaling  - radar power has huge range, log compresses it so CNN can see everything
    2. clip + normalize - cuts off noise spikes at top/bottom 1%, then scales to [0,1]
    3. colormap - converts grayscale to RGB using viridis (needed 3 channels for CNN)
    4. resize to 128x128
    """
    data = signal_matrix.astype(np.float32)

    # step 1: log scaling
    # without this, strong reflections completely dominate and weak ones vanish
    # adding epsilon so we never take log(0) which would give -inf
    if np.min(data) >= 0:
        data = 10.0 * np.log10(data + config.LOG_EPSILON)

    # step 2: percentile clipping then normalize to [0, 1]
    # i tried plain min-max first but one bad spike would crush everything else to near-zero
    # clipping at 1%-99% ignores outliers and gives a much better image
    p_low, p_high = config.CLIP_PERCENTILE
    vmin = np.percentile(data, p_low)
    vmax = np.percentile(data, p_high)

    if vmax > vmin:
        data = np.clip(data, vmin, vmax)
        normalized = (data - vmin) / (vmax - vmin)
    else:
        # completely flat signal edge case - just return a blank image
        normalized = np.zeros_like(data)

    # step 3: apply colormap to get RGB image
    # viridis output is RGBA so slice off the alpha channel with [:, :, :3]
    # i chose viridis because its perceptually uniform - equal data steps look equal to the eye
    # tried magma too but viridis gave cleaner separation when i visualized all 3 classes
    colormap_fn = getattr(cm, colormap_name, cm.viridis)
    rgb_float = colormap_fn(normalized)[:, :, :3]  # drop alpha
    rgb_uint8 = (rgb_float * 255.0).astype(np.uint8)

    # step 4: resize to target shape using PIL bicubic
    # 128x128 is small enough to train fast but large enough to keep the doppler patterns visible
    # bicubic is smoother than bilinear for upscaling
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    h, w = target_shape
    pil_resized = pil_img.resize((w, h), resample=Image.BICUBIC)

    return np.array(pil_resized, dtype=np.uint8)


def save_spectrogram_image(spectrogram_rgb, output_path):
    """Save an RGB numpy array as a PNG file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(spectrogram_rgb, mode="RGB")
    img.save(path, format="PNG")
    return path


def process_all_raw_signals(raw_dir=config.RAW_DATA_DIR, output_dir=config.SPECTROGRAM_DIR,
                             max_samples_per_class=None):
    """
    Go through all the raw CSV files in dataset/raw, convert each one to a spectrogram PNG,
    and save a manifest.json with paths + labels so the training script can find everything.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []

    for folder_name, class_name in config.RAW_FOLDER_TO_CLASS.items():
        class_id = config.CLASS_TO_ID[class_name]
        class_out_dir = out_path / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        # look in both possible locations for the raw data
        csv_files = []
        for search_dir in [raw_path / folder_name, raw_path / "data" / folder_name]:
            if search_dir.exists():
                csv_files.extend(search_dir.rglob("*.csv"))

        print(f"  found {len(csv_files)} CSV files for class '{class_name}'")

        if max_samples_per_class is not None:
            csv_files = csv_files[:max_samples_per_class]

        for idx, csv_file in enumerate(csv_files):
            try:
                rel_parts = csv_file.relative_to(raw_path).parts
                sample_name = "_".join(rel_parts).replace(".csv", ".png")
                png_path = class_out_dir / sample_name

                # skip if already processed - saves time when re-running
                if not png_path.exists():
                    raw_signal = read_raw_csv_signal(csv_file)
                    spectrogram = convert_signal_to_spectrogram_image(raw_signal)
                    save_spectrogram_image(spectrogram, png_path)

                manifest.append({
                    "sample_id":  f"{class_name}_{idx:05d}",
                    "class_name": class_name,
                    "class_id":   class_id,
                    "csv_path":   str(csv_file),
                    "image_path": str(png_path),
                })

            except Exception as err:
                logger.error(f"failed on {csv_file}: {err}")

    # save manifest so train.py knows where all the images are and what class they belong to
    manifest_path = out_path.parent / config.PROCESSED_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  done - {len(manifest)} spectrograms saved, manifest at {manifest_path}")
    return manifest
