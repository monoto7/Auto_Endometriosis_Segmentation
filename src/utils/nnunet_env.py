import os
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\Jasmin Repos\AES\Auto_Endometriosis_Segmentation")

NNUNET_BASE = Path(r"C:\Users\cooll\OneDrive\Documents\SPARC\Preprocessing\nnUnet")

NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"
NNUNET_EXPORTS = NNUNET_BASE / "evaluation_exports"

NNUNET_CUSTOM_TRAINERS = PROJECT_ROOT / "src" / "nnunet_custom_trainers"


def setup_nnunet_environment():
    """
    Set nnU-Net v2 environment variables from inside Python.

    This makes nnU-Net runnable from PyCharm without .bat files.
    """

    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)
    NNUNET_EXPORTS.mkdir(parents=True, exist_ok=True)
    NNUNET_CUSTOM_TRAINERS.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)

    # Allows nnU-Net to find nnUNetTrainer_100epochs outside site-packages.
    os.environ["nnUNet_extTrainer"] = str(NNUNET_CUSTOM_TRAINERS)

    # Also keep the project importable for our own scripts.
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    project_path = str(PROJECT_ROOT)

    if project_path not in old_pythonpath:
        os.environ["PYTHONPATH"] = project_path + os.pathsep + old_pythonpath
    else:
        os.environ["PYTHONPATH"] = old_pythonpath

    print("nnU-Net environment variables:")
    print(f"nnUNet_raw={os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed={os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results={os.environ['nnUNet_results']}")
    print(f"nnUNet_extTrainer={os.environ['nnUNet_extTrainer']}")
    print(f"PYTHONPATH={os.environ['PYTHONPATH']}")


def get_nnunet_environment():
    setup_nnunet_environment()
    return os.environ.copy()