"""Fetches the open-source LivePortrait engine and its pretrained weights.

    python setup_liveportrait.py            # clone + pip install + weights
    python setup_liveportrait.py --no-pip   # skip installing Python packages
    python setup_liveportrait.py --anime    # also fetch the anime engine (THA3)

LivePortrait (https://github.com/KwaiVGI/LivePortrait) is MIT licensed. It
uses InsightFace detection models, which are for non-commercial research use
only; replace them before any commercial use.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/KwaiVGI/LivePortrait.git"
# Pinned so the engine API this project calls into cannot drift underneath it.
COMMIT = "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"
WEIGHTS_REPO = "KlingTeam/LivePortrait"

# Anime engine: Talking Head Anime 3 (code MIT, models CC BY 4.0, Pramook Khungurn).
THA3_REPO = "https://github.com/pkhungurn/talking-head-anime-3-demo.git"
THA3_COMMIT = "8946939ec7b417f443d7b0e3fcd97384313fcdb8"
THA3_MODELS = "https://www.dropbox.com/s/y7b8jl4n2euv8xe/talking-head-anime-3-models.zip?dl=1"
# nagadomi/lbpcascade_animeface (no license declared upstream): optional,
# only used to place the head automatically.
ANIME_FACE_CASCADE = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "LivePortrait"


def run(*cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)), cwd=cwd)


def clone():
    if (VENDOR / ".git").exists():
        print(f"LivePortrait already at {VENDOR}")
    else:
        VENDOR.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", REPO, VENDOR)
    run("git", "fetch", "--depth", "1", "origin", COMMIT, cwd=VENDOR)
    run("git", "checkout", "--quiet", COMMIT, cwd=VENDOR)


def install_torch(gpu: bool):
    try:
        import torch  # noqa: F401

        print(f"PyTorch {torch.__version__} already installed")
        return
    except ImportError:
        pass
    index = "https://download.pytorch.org/whl/cu121" if gpu else "https://download.pytorch.org/whl/cpu"
    run(sys.executable, "-m", "pip", "install", "torch==2.3.0", "torchvision==0.18.0",
        "--index-url", index, "--extra-index-url", "https://pypi.org/simple")


def install_packages():
    reqs = "requirements_macOS.txt" if sys.platform == "darwin" else "requirements.txt"
    gpu = shutil.which("nvidia-smi") is not None
    if sys.platform == "darwin":
        run(sys.executable, "-m", "pip", "install", "-r", VENDOR / reqs)
        run(sys.executable, "-m", "pip", "install", "-r", ROOT / "requirements.txt")
        return
    install_torch(gpu)
    if not gpu:
        print("No NVIDIA GPU found: installing CPU onnxruntime (rendering will be slow).")
        run(sys.executable, "-m", "pip", "install", "-r", VENDOR / "requirements_base.txt", "onnxruntime")
    else:
        run(sys.executable, "-m", "pip", "install", "-r", VENDOR / reqs)
    run(sys.executable, "-m", "pip", "install", "-r", ROOT / "requirements.txt")


def download_weights():
    from huggingface_hub import snapshot_download

    dest = VENDOR / "pretrained_weights"
    snapshot_download(
        repo_id=WEIGHTS_REPO,
        local_dir=dest,
        allow_patterns=["insightface/**", "liveportrait/**"],
    )
    needed = [
        "liveportrait/base_models/appearance_feature_extractor.pth",
        "liveportrait/base_models/motion_extractor.pth",
        "liveportrait/base_models/spade_generator.pth",
        "liveportrait/base_models/warping_module.pth",
        "liveportrait/retargeting_models/stitching_retargeting_module.pth",
        "liveportrait/landmark.onnx",
    ]
    missing = [p for p in needed if not (dest / p).exists()]
    if missing:
        sys.exit(f"Weights incomplete, missing: {missing}")
    print(f"Weights ready in {dest}")


def setup_anime():
    import urllib.request
    import zipfile

    tha = ROOT / "vendor" / "talking-head-anime-3-demo"
    if not (tha / ".git").exists():
        run("git", "clone", THA3_REPO, tha)
    run("git", "fetch", "--depth", "1", "origin", THA3_COMMIT, cwd=tha)
    run("git", "checkout", "--quiet", THA3_COMMIT, cwd=tha)
    models = tha / "data" / "models"
    if not (models / "separable_float" / "editor.pt").exists():
        zpath = ROOT / "vendor" / "tha3-models.zip"
        print("Downloading Talking Head Anime 3 models (~830 MB)…", flush=True)
        urllib.request.urlretrieve(THA3_MODELS, zpath)
        models.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(models)
        zpath.unlink()
    cascade = ROOT / "vendor" / "lbpcascade_animeface.xml"
    if not cascade.exists():
        try:
            urllib.request.urlretrieve(ANIME_FACE_CASCADE, cascade)
        except OSError as e:
            print(f"Anime face detector unavailable ({e}); head placement falls back to the silhouette.")
    sr = ROOT / "vendor" / "models" / "realesr-animevideov3.pth"
    if not sr.exists():
        sr.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", sr)
    run(sys.executable, "-m", "pip", "install", "psd-tools")
    print(f"Anime engine ready in {tha}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pip", action="store_true", help="don't install Python packages")
    ap.add_argument("--no-weights", action="store_true", help="don't download model weights")
    ap.add_argument("--anime", action="store_true", help="also set up the anime engine (Talking Head Anime 3)")
    args = ap.parse_args()
    if not args.no_pip and not ((3, 9) <= sys.version_info[:2] <= (3, 11)):
        sys.exit("LivePortrait's pinned dependencies need Python 3.9-3.11 (3.10 recommended), e.g.\n"
                 "  conda create -n memoire python=3.10 && conda activate memoire")
    clone()
    if not args.no_pip:
        install_packages()
    if not args.no_weights:
        download_weights()
    if args.anime:
        setup_anime()
    print("\nDone. Start the rigger with:  python -m memoire_rig serve")


if __name__ == "__main__":
    main()
