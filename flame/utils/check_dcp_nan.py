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
    # tokenizer: str,
    # hf_ckpt_dir: Optional[str]
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)
    # hf_ckpt_dir = os.path.join(path, f'hf_checkpoint/step-{step}') if hf_ckpt_dir is None else hf_ckpt_dir  # defualt huggingface checkpoint directory
    # logger.info(f"Saving the config to {hf_ckpt_dir}")
    # config.save_pretrained(hf_ckpt_dir)
    # logger.info(f"Loading the tokenizer from {tokenizer}")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    # logger.info(f"Saving the tokenizer to {hf_ckpt_dir}")
    # tokenizer.save_pretrained(hf_ckpt_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config)
        logger.info(model)
        logger.info("Loading state dict from the checkpoint")

        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        # torch.load now with default weights_only=True will work
        # model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model'])

        #check NaN
        logger.info(f"Check NaN/Inf for state_dict from torch load after dcp_to_torch_save")
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
        nan_keys = []
        inf_keys = []
        for key, tensor in state_dict.items():
            if torch.is_floating_point(tensor) and torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                nan_keys.append((key, nan_count, tensor.numel()))
                logger.warning(f"NaN detected in {key}: {nan_count}/{tensor.numel()} values")
            elif torch.is_floating_point(tensor) and torch.isinf(tensor).any():
                inf_count = torch.isinf(tensor).sum().item()
                inf_keys.append((key, inf_count, tensor.numel()))
                logger.warning(f"Inf detected in {key}: {inf_count}/{tensor.numel()} values")
        if nan_keys:
            logger.error(f"Found NaN in {len(nan_keys)} parameters!")
            # Optionally raise or continue
            # raise ValueError(f"NaN found in {len(nan_keys)} keys: {[k for k,_,_ in nan_keys]}")
        else:
            logger.info("No NaN values found in checkpoint.")
        if inf_keys:
            logger.error(f"Found Inf in {len(inf_keys)} parameters!")
        else:
            logger.info("No Inf values found in checkpoint.")


        logger.info(f"Check NaN/Inf for torch model after load_state_dict")
        model.load_state_dict(state_dict)
        # model = model.cuda()
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                print(f"!!! NaN/Inf in {name}: nan={nan_count}, inf={inf_count}")
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda") # input_ids 也要在同一设备上
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits
            print(f"Logits shape: {logits.shape}")
            print(f"Logits has NaN: {torch.isnan(logits).any()}")
            print(f"Logits sample: {logits[0, -1, :10]}")
            print(f"Logits std: {logits.std()}")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--tokenizer", type=str, required=True)
    # parser.add_argument("--hf-ckpt-dir", type=str, default=None)
    args = parser.parse_args()
    # save_pretrained(args.path, args.step, args.config, args.tokenizer, args.hf_ckpt_dir)
    save_pretrained(args.path, args.step, args.config)