"""Execution settings for the original H100 and the optional smaller CUDA profile."""
import os


PROFILE = os.environ.get("CRISM_GPU_PROFILE", "h100")
if PROFILE not in ("h100", "portable"):
    raise ValueError("CRISM_GPU_PROFILE must be h100 or portable")


def inference_batches(config):
    if PROFILE == "portable":
        config["lwnet"]["batch_size"] = min(config["lwnet"]["batch_size"], 4)
        config["n2n4m"]["batch_size"] = min(config["n2n4m"]["batch_size"], 16)
