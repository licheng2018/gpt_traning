#!/usr/bin/env python3
# ============================================================
# Day 4 — Train GPT-Neo-1.3B (Practice) (SKELETON, NO SOLUTION)
# Training Train GPT-Neo-1.3B with check points
# ============================================================

import os
import time
import math
import argparse
from dataclasses import dataclass
import datasets
from typing import Dict, Any, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional: FSDP (PyTorch native)
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
except Exception:
    FSDP = None  # will assert if user chooses fsdp

# HF stack
from transformers import AutoTokenizer, AutoModelForCausalLM

from functools import partial
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

from contextlib import nullcontext


# -------------------------
# Args / Config
# -------------------------
@dataclass
class TrainCfg:
    model_name: str
    seq_len: int
    microbatch: int
    grad_accum: int
    steps: int
    warmup_steps: int
    lr: float
    weight_decay: float
    max_grad_norm: float
    strategy: str          # "ddp" or "fsdp"
    fp16: bool
    seed: int
    log_every: int
    save_every: int
    out_dir: str
    dataset: str           # "wikitext" or "synthetic"
    num_workers: int
    checkpoint: bool


def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser()

    # Core
    p.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--microbatch", type=int, default=1)     # per-GPU batch
    p.add_argument("--grad_accum", type=int, default=16)    # accum steps
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=50)

    # Optim
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Dist
    p.add_argument("--strategy", type=str, choices=["ddp", "fsdp"], default="ddp")

    # Precision
    p.add_argument("--fp16", action="store_true")  # On T4 you likely want --fp16

    # Data
    p.add_argument("--dataset", type=str, choices=["wikitext", "synthetic"], default="wikitext")
    p.add_argument("--num_workers", type=int, default=2)

    # Misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=0)  # 0 disables saving
    p.add_argument("--out_dir", type=str, default="./day3_out")

    #check point
    p.add_argument("--checkpoint", action="store_true")

    a = p.parse_args()
    return TrainCfg(**vars(a))


# -------------------------
# Dist utilities
# -------------------------
def ddp_init() -> Tuple[int, int, int, torch.device]:
    """
    Initialize torch.distributed from torchrun env vars.
    Returns: (rank, local_rank, world_size, device)
    """
    # TODO: read rank/local_rank/world_size from env (torchrun sets these)
    # - RANK
    # - LOCAL_RANK
    # - WORLD_SIZE
    #
    # TODO: dist.init_process_group(backend="nccl")
    #
    # TODO: set CUDA device to local_rank
    #
    # TODO: return them
    # raise NotImplementedError
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    
      
    
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    
        device = torch.device("cuda", local_rank)
    else:
        # single-process fallback for notebook debugging
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda", 0 if torch.cuda.is_available() else "cpu")
    return rank, local_rank, world_size, device
    

def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int, rank: int) -> None:
    s = seed + rank
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# -------------------------
# Data (skeleton)
# -------------------------
class SeqBatcher:
    """
    Produces (input_ids, labels) of shape [microbatch, seq_len]
    You can implement either:
      - real dataset (wikitext-103) token stream -> chunk into seq_len
      - synthetic random tokens fallback

    For Day3, correctness > sophistication.
    """
    def __init__(self, tokenizer, cfg: TrainCfg, device: torch.device, rank: int, world_size: int):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.rank = rank
        self.world_size = world_size

        # TODO: build token stream for dataset
        # If cfg.dataset == "wikitext":
        #   - use datasets.load_dataset("wikitext", "wikitext-103-raw-v1")
        #   - concatenate train texts
        #   - tokenize into a single 1D tensor of token ids
        #   - shard by rank to avoid duplicates
        #
        # Else "synthetic":
        #   - create a 1D tensor with random tokens in vocab range
        #
        # Store token_stream: 1D LongTensor on CPU (or pinned memory)
        if cfg.dataset == "wikitext":
            # 1) load raw train split
            ds = datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            
            # 2) concatenate all text into one long string
            texts = ds["text"]
            joined_text = "\n\n".join(texts)

            # 3) tokenize into one long 1D token stream
            token_ids = tokenizer(joined_text, return_tensors = "pt", add_special_tokens = False
                                  )["input_ids"][0].to(torch.long)
            
            # 4) shard by rank to reduce duplicate data across processes
            token_ids = token_ids[self.rank::self.world_size]

            self.token_stream = token_ids.cpu()

        elif cfg.dataset == "synthetic":
            # build a long enough fake token stream
            vocab_size = tokenizer.vocab_size

            # rough lower bound: enough for all optimizer steps and accum steps
            total_tokens = (cfg.steps 
                            * cfg.grad_accum 
                            * cfg.microbatch  
                            * (cfg.seq_len + 1) * 2)

            token_ids = torch.randint(low = 0, 
                                      high = vocab_size, 
                                      size = (total_tokens,), 
                                      dtype = torch.long)

            #shard synthetic data too, for consistency
            token_ids = token_ids[self.rank::self.world_size]
            self.token_stream = token_ids.cpu()

        else:
            raise ValueError(f"Unsupported dataset: {cfg.dataset}")

        self.pos = 0

        # basic sanity check
        min_needed = self.cfg.microbatch * (self.cfg.seq_len + 1)
        if len(self.token_stream) < min_needed:
            raise ValueError(
                f"token_stream too short: got {len(self.token_stream)} tokens, "
                f"but need at least {min_needed} for one batch."
            )
        
            
        

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          input_ids: [B, T]
          labels:    [B, T]
        """
        B = self.cfg.microbatch
        T = self.cfg.seq_len

        # TODO: slice token_stream into B*T+1 tokens and reshape into [B, T+1]
        # Then:
        #   input_ids = chunk[:, :-1]
        #   labels    = chunk[:, 1:]
        #
        # TODO: move to device (non_blocking)
        # raise NotImplementedError

        # how many tokens so that label can shift one
        need = B * (T + 1)

        # If remaining tokens are not enough for one batch, wrap around
        if self.pos + need > len(self.token_stream):
            self.pos = 0

        #slice a contigous chunk and reshape to [B, T + 1]
        chunk = self.token_stream[self.pos: self.pos + need]
        chunk = chunk.view(B, T + 1)

        # Build causal LM inputs/label
        input_ids = chunk[:,:-1]
        labels = chunk[:,1:]

        # Advance read pointer
        self.pos += need

        # move data to GPU
        input_ids = input_ids.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        return input_ids, labels

# -------------------------
# Model wrapping
# -------------------------
def build_model_and_tokenizer(cfg: TrainCfg, device: torch.device):
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    # GPT-Neo often has no pad token; for causal LM you can set pad to eos for convenience
    if tok.pad_token is None:
        # TODO: set tok.pad_token = tok.eos_token
        # pass
        # tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        tok.pad_token = tok.eos_token



    # Model
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.train()

    # TODO: For training, disable cache:
    # model.config.use_cache = False
    model.config.use_cache = False

    # check point
    if cfg.checkpoint:
        model.gradient_checkpointing_enable()

    # TODO: move model to device for DDP; for FSDP you may still move first
    # model.to(device)
    model.to(device)



    return model, tok


def wrap_ddp(model: torch.nn.Module, device: torch.device, local_rank: int) -> torch.nn.Module:
    # TODO: DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    # raise NotImplementedError
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                find_unused_parameters=False)
    return model


def wrap_fsdp(model: torch.nn.Module, cfg: TrainCfg) -> torch.nn.Module:
    """
    FSDP full-shard baseline for Day3.
    You implement:
      - auto_wrap_policy over transformer blocks
      - MixedPrecision fp16 on T4 (param/reduce/buffer)
      - ShardingStrategy.FULL_SHARD
      - use_orig_params=True recommended (if available)
    """
    assert FSDP is not None, "FSDP not available in this environment."

    # TODO: define which module class corresponds to a "transformer block" for GPT-Neo
    # Hint: print(model) and locate block class name; often something like GPTNeoBlock
    # transformer_block_cls = None  # TODO
    transformer_block_cls = GPTNeoBlock


    # TODO: build auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={transformer_block_cls})
    # auto_wrap = None  # TODO
    auto_wrap = partial(transformer_auto_wrap_policy,
                        transformer_layer_cls={transformer_block_cls},)

    mp = None
    if cfg.fp16:
        # TODO: MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
        # pass
        mp = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # TODO: return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=auto_wrap, mixed_precision=mp, ...)
    # raise NotImplementedError
    model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap,
    mixed_precision=mp,
    use_orig_params=True,
    )

    return model


# -------------------------
# Optim / AMP
# -------------------------
def build_optimizer(model: torch.nn.Module, cfg: TrainCfg) -> torch.optim.Optimizer:
    # TODO: AdamW over model.parameters(), lr, weight_decay
    # raise NotImplementedError
    return torch.optim.AdamW(model.parameters(),
                             lr = cfg.lr,
                             weight_decay=cfg.weight_decay,)


def maybe_autocast(cfg: TrainCfg):
    # TODO: return torch.cuda.amp.autocast(dtype=torch.float16) if cfg.fp16 else nullcontext()
    # raise NotImplementedError
    if cfg.fp16:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(cfg: TrainCfg) -> Optional[torch.cuda.amp.GradScaler]:
    if cfg.fp16:
        # TODO: return torch.cuda.amp.GradScaler()
        return torch.cuda.amp.GradScaler() 
    return None


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def get_mem_gb() -> Dict[str, float]:
    # Note: allocated/reserved are per-process
    alloc = torch.cuda.max_memory_allocated() / (1024**3)
    rsvd  = torch.cuda.max_memory_reserved() / (1024**3)
    return {"max_alloc_gb": alloc, "max_reserved_gb": rsvd}


def allreduce_float(x: float, device: torch.device) -> float:
    """
    Average a scalar across ranks.
    """
    t = torch.tensor([x], device=device, dtype=torch.float32)
    # TODO: dist.all_reduce(t, op=dist.ReduceOp.SUM); t /= world_size
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    t /= world_size

    return t.item()


# -------------------------
# Train loop
# -------------------------
def train(cfg: TrainCfg):
    rank, local_rank, world_size, device = ddp_init()
    set_seed(cfg.seed, rank)

    os.makedirs(cfg.out_dir, exist_ok=True)
    if is_main(rank):
        print(f"[Day3] cfg={cfg}")
        print(f"[Day3] rank={rank} local_rank={local_rank} world_size={world_size} device={device}")

    # Build model/tokenizer
    model, tok = build_model_and_tokenizer(cfg, device)

    # Wrap distributed
    if cfg.strategy == "ddp":
        model = wrap_ddp(model, device, local_rank)
    elif cfg.strategy == "fsdp":
        model = wrap_fsdp(model, cfg)
    else:
        raise ValueError(cfg.strategy)

    # Optim
    opt = build_optimizer(model, cfg)
    scaler = build_grad_scaler(cfg)

    # Data
    batcher = SeqBatcher(tok, cfg, device, rank, world_size)

    # Reset peak mem stats
    torch.cuda.reset_peak_memory_stats()

    # Throughput tracking
    step_times = []
    tokens_per_step = cfg.microbatch * cfg.seq_len * world_size * cfg.grad_accum # tokens processed per step (not counting labels shift)
    # NOTE: if you define "tokens" differently, be consistent.

    # Main loop
    model.train()
    t0_global = time.time()

    for step in range(1, cfg.steps + 1):
        t_step0 = time.time()

        # Gradient accumulation
        # TODO: opt.zero_grad(set_to_none=True) at start of "logical step"
        # Then for i in range(cfg.grad_accum):
        #   - get batch
        #   - forward under autocast
        #   - loss = model(...).loss
        #   - scale loss by 1/grad_accum
        #   - backward (with scaler if fp16)
        #
        # TODO: after accumulation:
        #   - clip grad (handle scaler case)
        #   - optimizer step (handle scaler)
        #   - scaler update if needed
        # raise NotImplementedError
        
        #one optimizer step starts here
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for i in range(cfg.grad_accum):
            input_ids, labels = batcher.next_batch()

            # 只有 DDP 且不是最后一个 accumulation step 时，关闭梯度同步
            use_no_sync = (
                cfg.strategy == "ddp"
                and hasattr(model, "no_sync")
                and i < cfg.grad_accum - 1
            )

            sync_ctx = model.no_sync() if use_no_sync else nullcontext()

            with sync_ctx:
                with maybe_autocast(cfg):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    loss_sum += loss.detach().float().item()
                    loss = loss / cfg.grad_accum
            
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

        # Gradient clipping + optimizer step
        if scaler is not None:
            # unscale before clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()

        # Timing
        t_step = time.time() - t_step0

        # Skip warmup for avg stats
        if step > cfg.warmup_steps:
            step_times.append(t_step)

        # Logging
        if (step % cfg.log_every == 0) or (step == 1):
            # TODO: compute avg_step_ms over recent window or global
            # TODO: compute tokens/s using avg_step_time and tokens_per_step * grad_accum? (define clearly)
            # Hint: "effective tokens per optimizer step" includes grad_accum.
            #
            # TODO: optionally allreduce avg over ranks to report single number
            #
            # Also print memory stats (max_alloc/reserved)
            avg_loss_local = loss_sum / cfg.grad_accum
            avg_loss = allreduce_float(avg_loss_local, device)

            if len(step_times) > 0:
                avg_step_s_local = sum(step_times) / len(step_times)
            else:
                avg_step_s_local = t_step
            
            avg_step_s = allreduce_float(avg_step_s_local, device)
            toks_per_s = tokens_per_step / avg_step_s if avg_step_s > 0 else float("nan")

            if is_main(rank):
                mem = get_mem_gb()
                print(
                    f"[step {step:04d}] "
                    f"loss={avg_loss:.4f} "
                    f"step_s={avg_step_s:.4f} "
                    f"tok/s={toks_per_s:.2f} "
                    f"mem={mem}"
                )

        # Optional checkpoint
        if cfg.save_every > 0 and (step % cfg.save_every == 0):
            if is_main(rank):
                ckpt_dir = os.path.join(cfg.out_dir, f"step_{step:04d}")
                os.makedirs(ckpt_dir, exist_ok=True)

                # unwrap DDP/FSDP if possible
                save_model = model.module if hasattr(model, "module") else model

                # save HF model/tokenizer if available
                if hasattr(save_model, "save_pretrained"):
                    save_model.save_pretrained(ckpt_dir)
                if hasattr(tok, "save_pretrained"):
                    tok.save_pretrained(ckpt_dir)

    # Final report
    elapsed = time.time() - t0_global
    if len(step_times) > 0:
        avg_step = sum(step_times) / len(step_times)
    else:
        avg_step = float("nan")

    if is_main(rank):
        mem = get_mem_gb()
        print(f"[DONE] elapsed_s={elapsed:.1f} avg_step_s={avg_step:.4f} mem={mem}")

    # Cleanup
    # TODO: dist.barrier(); dist.destroy_process_group()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()