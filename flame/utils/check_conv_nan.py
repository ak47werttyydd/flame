# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta
from typing import Optional

import fla  # noqa
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# import custom_models


@torch.inference_mode()
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str,
    hf_ckpt_dir: Optional[str]
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)
    hf_ckpt_dir = os.path.join(path, f'hf_checkpoint/step-{step}') if hf_ckpt_dir is None else hf_ckpt_dir  # defualt huggingface checkpoint directory
    logger.info(f"Saving the config to {hf_ckpt_dir}")
    config.save_pretrained(hf_ckpt_dir)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {hf_ckpt_dir}")
    tokenizer.save_pretrained(hf_ckpt_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the dcp checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        # print A_log loaded from DCP
        for name, param in model.named_parameters():
            if 'A_log' in name:
                print(f"[before save] {name}: {param.data}")
                if torch.isnan(param).any():
                    print(f"  !!! NaN introduced by save/load cycle")
        
        logger.info(f"Saving the model to {hf_ckpt_dir}")
        model._tied_weights_keys = {}
        model.save_pretrained(hf_ckpt_dir)

        # print A_log loaded from HF ( DCP -> HF )
        model2 = AutoModelForCausalLM.from_pretrained(hf_ckpt_dir, trust_remote_code=True)
        for name, param in model2.named_parameters():
            if 'A_log' in name:
                print(f"[after reload] {name}: {param.data}")
                if torch.isnan(param).any():
                    print(f"  !!! NaN introduced by save/load cycle")

        
if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--hf-ckpt-dir", type=str, default=None)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer, args.hf_ckpt_dir)
    # save_pretrained(args.path, args.step, args.config)