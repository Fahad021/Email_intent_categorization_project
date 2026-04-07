from __future__ import annotations

import logging
import os
import time

from llama_cpp import Llama

from .config import Config


def load_model(cfg: Config, log: logging.Logger) -> Llama:
    """
    Load a GGUF model applying llama.cpp best practices:
      - n_threads = physical cores (not logical) — passed via cfg
      - use_mmap  = True  — fast load via memory mapping
      - use_mlock = cfg.use_mlock  — optional: pin weights in RAM
      - flash_attn= cfg.flash_attn — optional: faster on AVX2/AVX512
      - last_n_tokens_size = n_ctx — KV-cache prefix reuse
      - add_bos = False — caller adds <s> manually for Mistral v0.2
    """
    t0 = time.time()
    log.info(
        "model_load_start",
        extra={
            "file_in":    cfg.model_path,
            "n_threads":  cfg.n_threads,
            "use_mlock":  cfg.use_mlock,
            "flash_attn": cfg.flash_attn,
            "n_ctx":      cfg.n_ctx,
        },
    )

    model = Llama(
        model_path         = cfg.model_path,
        n_ctx              = cfg.n_ctx,
        n_gpu_layers       = cfg.n_gpu_layers,
        n_threads          = cfg.n_threads,
        n_batch            = cfg.n_batch,
        last_n_tokens_size = cfg.n_ctx,
        use_mmap           = True,
        use_mlock          = cfg.use_mlock,
        flash_attn         = cfg.flash_attn,
        add_bos            = False,
        verbose            = False,
    )

    log.info(
        "model_load_complete",
        extra={
            "model":        os.path.basename(cfg.model_path),
            "model_ctx":    cfg.n_ctx,
            "n_threads":    cfg.n_threads,
            "n_batch":      cfg.n_batch,
            "n_gpu_layers": cfg.n_gpu_layers,
            "use_mlock":    cfg.use_mlock,
            "flash_attn":   cfg.flash_attn,
            "latency_ms":   int((time.time() - t0) * 1000),
        },
    )
    return model
