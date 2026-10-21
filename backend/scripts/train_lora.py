"""Fine-tune Phi-3-mini (or any causal LM) with LoRA to explain fund recommendations.

    cd backend
    python scripts/train_lora.py     # Phi-3-mini; saves to artifacts/lora-phi-3-mini-4k-instruct

The base model comes from --base-model, else MF_BASE_MODEL, else microsoft/Phi-3-mini-4k-instruct.
The adapter is saved under artifacts/ and the API picks it up automatically (only for the
same base model). Defaults fit a free Colab T4: batch 2 x 8 accumulation steps, gradient
checkpointing on, float16.

For chat models such as Phi-3-mini each example is written in the model's own chat format
(<|user|> prompt <|end|> <|assistant|> answer <|end|>), exactly as the API will prompt it,
and the LoRA layers are chosen for the architecture (Phi-3: qkv_proj, o_proj).

Fixes over the original notebook's training cell:
  * Loss only on the answer. The notebook trained on prompt + answer, so most of the
    gradient went into memorising the prompt layout.
  * The answer ends with an EOS token that is *not* masked out. The notebook used
    pad_token = eos_token with DataCollatorForLanguageModeling, which sets every EOS label
    to -100, so the model never learned to stop and rambled until max_new_tokens.
  * Dynamic padding instead of padding every example to 512 tokens.
  * Targets come from the fund's real numbers compared with its peers
    (explanations.describe_fund), not one sentence claiming "steady returns ... long-term
    growth" for every fund, liquid and negative-return funds included.
  * Goals vary per example, and a held-out set of *funds* is used for evaluation, scored
    on eval loss and on how often generations only cite percentages that are in the facts.
  * Prompts come from prompts.EXPLAIN_TEMPLATE, the same template the API uses.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make mf_recommender importable

from mf_recommender.categories import profile_for  # noqa: E402
from mf_recommender.config import Settings, default_adapter_dir  # noqa: E402
from mf_recommender.data import RISK_BANDS, risk_band_for_step  # noqa: E402
from mf_recommender.explanations import describe_fund, prompt_inputs  # noqa: E402
from mf_recommender.model_utils import (  # noqa: E402
    has_chat_template,
    lora_target_modules,
    to_chat_example,
    to_chat_prompt,
)
from mf_recommender.prompts import EXPLAIN_TEMPLATE  # noqa: E402

GOALS = {
    "short": [
        "Park my surplus cash safely for {m} months",
        "I need somewhere safe for my emergency fund that I can withdraw quickly",
        "Keep money I need in {m} months safe with stable returns",
    ],
    "medium": [
        "Steady income with low volatility over {y} years",
        "Grow my savings moderately over the next {y} years without big swings",
        "I want better returns than a fixed deposit over {y} years",
    ],
    "long": [
        "Long-term wealth creation over {y} years",
        "Build a retirement corpus over {y} years",
        "Save tax and grow my money for {y} years",
        "High growth over {y} years, I can handle volatility",
        "Save for my child's education in {y} years",
    ],
}
_RANGES = {"short": (1, 9), "medium": (2, 4), "long": (5, 15)}  # months for short, years otherwise


def horizon_group(category: str) -> str:
    p = profile_for(category)
    if p.max_years is not None and p.max_years <= 2:
        return "short"
    return "medium" if p.min_years < 5 else "long"


def sample_goal(rng: random.Random, group: str) -> tuple[str, float]:
    template = rng.choice(GOALS[group])
    lo, hi = _RANGES[group]
    n = rng.randint(lo, hi)
    if group == "short":
        return template.format(m=n), n / 12
    return template.format(y=n), float(n)


def sample_requested_risk(rng: random.Random, fund: dict, mismatch_rate: float) -> str | None:
    """The risk the investor picked in the UI. Usually the fund's own band (the API filters
    by it); sometimes none; sometimes another band, as when no fund in the chosen category
    has the requested risk and the API shows the closest ones."""
    own = risk_band_for_step(fund["risk_step"])
    roll = rng.random()
    if roll < 0.15:
        return None
    if roll < 0.15 + mismatch_rate:
        return rng.choice([b for b in RISK_BANDS if b != own])
    return own


def build_examples(funds, *, per_fund: int, mismatch_rate: float, seed: int) -> list[dict]:
    """Prompt/response pairs. A share of goals (and risk choices) deliberately mismatch the
    fund so the model also learns to say when a fund is *not* a good fit."""
    rng = random.Random(seed)
    from mf_recommender.ranking import parse_horizon_years

    examples = []
    for fund in funds:
        own = horizon_group(fund["category"])
        for _ in range(per_fund):
            group = own
            if rng.random() < mismatch_rate:
                group = rng.choice([g for g in GOALS if g != own])
            goal, _ = sample_goal(rng, group)
            risk = sample_requested_risk(rng, fund, mismatch_rate)
            # Use the same parser as the API so training and inference see identical inputs.
            horizon = parse_horizon_years(goal)
            examples.append(
                {
                    "scheme_code": int(fund["scheme_code"]),
                    "prompt": EXPLAIN_TEMPLATE.format(**prompt_inputs(fund, goal, horizon, risk)),
                    "response": describe_fund(fund, goal, horizon, risk),
                }
            )
    return examples


def split_by_fund(examples: list[dict], eval_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    codes = sorted({e["scheme_code"] for e in examples})
    random.Random(seed).shuffle(codes)
    held_out = set(codes[: max(1, int(len(codes) * eval_fraction))])
    train = [e for e in examples if e["scheme_code"] not in held_out]
    evals = [e for e in examples if e["scheme_code"] in held_out]
    return train, evals


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict | None:
    """input_ids / labels for one example; labels = -100 (ignored) on the prompt, so the
    loss is on the answer and its end-of-turn token only."""
    if has_chat_template(tokenizer):
        # Chat model: tokenize the whole conversation once and mask the prompt part, which
        # ends with '<|assistant|>\n'. Tokenized exactly as the API's pipeline will (BOS included).
        prompt_text, full_text = to_chat_example(tokenizer, example["prompt"], example["response"])
        prompt_ids = tokenizer(prompt_text)["input_ids"]
        full_ids = tokenizer(full_text)["input_ids"]
        if full_ids[: len(prompt_ids)] == prompt_ids:
            answer_ids = full_ids[len(prompt_ids):]
        else:  # template tokenizes differently as a whole: tokenize the answer part alone
            answer_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    else:
        # Base model (e.g. Phi-2): prompt ends with "Answer:", the answer follows, then EOS.
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(" " + example["response"], add_special_tokens=False)["input_ids"]
    if not answer_ids or answer_ids[-1] != tokenizer.eos_token_id:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None  # rather than silently cutting off the answer or its EOS
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(Settings.from_env().data_path), help="Fund CSV (default: MF_DATA_PATH or data/)")
    p.add_argument("--base-model", default=Settings.from_env().base_model,
                   help="Default: MF_BASE_MODEL or microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--output", default=None,
                   help="Where to save the adapter (default: artifacts/lora-<model>, picked up automatically)")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2, help="Per step; 2 fits Phi-3-mini on a 16 GB T4")
    p.add_argument("--grad-accum", type=int, default=8, help="Steps per update (2 x 8 = 16 examples)")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--target-modules", default="auto",
        help='"auto" (by architecture; Phi-3: qkv_proj,o_proj), "all-linear", or comma-separated names',
    )
    p.add_argument("--examples-per-fund", type=int, default=3)
    p.add_argument("--mismatch-rate", type=float, default=0.15)
    p.add_argument("--eval-fraction", type=float, default=0.1)
    p.add_argument("--eval-generate", type=int, default=20, help="Held-out prompts to generate for the grounding check")
    p.add_argument("--max-funds", type=int, default=None, help="Use only N funds (smoke tests)")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True,
                   help="Recompute activations to save memory (default on; --no-gradient-checkpointing to disable)")
    p.add_argument("--save-jsonl", default=None, help="Also write the generated examples here")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output = args.output or str(default_adapter_dir(args.base_model))
    print(f"Base model: {args.base_model}\nAdapter will be saved to: {args.output}")

    import torch
    import transformers
    from datasets import Dataset
    from mf_recommender.chains import clean_generation, is_usable_explanation
    from mf_recommender.data import load_funds
    from packaging.version import Version
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    )

    transformers.set_seed(args.seed)
    funds = load_funds(args.data).to_dict("records")
    if args.max_funds:
        funds = funds[: args.max_funds]
    examples = build_examples(funds, per_fund=args.examples_per_fund, mismatch_rate=args.mismatch_rate, seed=args.seed)
    train_ex, eval_ex = split_by_fund(examples, args.eval_fraction, args.seed)
    if args.save_jsonl:
        with open(args.save_jsonl, "w") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{len(train_ex)} training / {len(eval_ex)} held-out examples from {len(funds)} funds")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe now: padding is masked via labels=-100, EOS is not
    tokenizer.padding_side = "right"
    print("Chat format:", "yes (prompts wrapped in the model's chat template)" if has_chat_template(tokenizer) else "no")

    def to_dataset(rows):
        tokenized = [t for t in (tokenize_example(r, tokenizer, args.max_length) for r in rows) if t]
        dropped = len(rows) - len(tokenized)
        if dropped:
            print(f"Skipped {dropped} examples longer than {args.max_length} tokens")
        return Dataset.from_list(tokenized)

    train_ds, eval_ds = to_dataset(train_ex), to_dataset(eval_ex)

    cuda = torch.cuda.is_available()
    # bfloat16 only on GPUs that run it natively (compute capability 8+: A100, L4, ...).
    # torch.cuda.is_bf16_supported() also says True on a T4, where bfloat16 is only emulated
    # and training ran about 8x slower (68 s per step on a T4); a T4 uses float16.
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    print(f"Precision: {str(dtype).replace('torch.', '')}" + (f" on {torch.cuda.get_device_name()}" if cuda else ""))
    dtype_kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="auto" if cuda else None, **{dtype_kw: dtype}
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False  # incompatible with checkpointing during training

    if args.target_modules == "auto":
        targets = lora_target_modules(model.config.model_type)
    elif args.target_modules == "all-linear":
        targets = "all-linear"
    else:
        targets = [t.strip() for t in args.target_modules.split(",")]
    print(f"LoRA layers ({model.config.model_type}): {targets}")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # warmup as a step count: `warmup_ratio` was removed in transformers 5.
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(0.05 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=str(Path(args.output) / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=bf16,
        fp16=cuda and not bf16,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True),
        processing_class=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Held-out eval loss: {metrics['eval_loss']:.4f}")

    out = Path(args.output)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"Adapter saved to {out}")
    # The best checkpoint is already loaded and saved above; drop the checkpoint folder (optimizer
    # state etc.) so it doesn't fill up Google Drive.
    shutil.rmtree(out / "checkpoints", ignore_errors=True)

    # Grounding check on held-out funds: do generations only cite numbers from the facts?
    model.eval()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    sample = eval_ex[: args.eval_generate]
    fund_by_code = {int(f["scheme_code"]): f for f in funds}
    grounded, lengths, generations = 0, [], []
    for ex in sample:
        # Prompted exactly as the API prompts it (chat-wrapped for Phi-3-mini).
        enc = tokenizer(to_chat_prompt(tokenizer, ex["prompt"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = clean_generation(tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        # The same quality gate the API applies, including the cost/returns contradiction check.
        grounded += int(is_usable_explanation(text, {"prompt": ex["prompt"]}, fund_by_code[ex["scheme_code"]]))
        lengths.append(len(text.split()))
        generations.append(text)
    if sample:
        print(
            f"Usable, grounded generations: {grounded}/{len(sample)}; "
            f"mean length {sum(lengths) / len(lengths):.0f} words"
        )
        print(f"\nExample prompt:\n{sample[0]['prompt']}\nGenerated: {generations[0]}\nTarget: {sample[0]['response']}")
    (out / "training_metrics.json").write_text(
        json.dumps(
            {"base_model": args.base_model, "eval_loss": metrics["eval_loss"], "grounded": grounded, "checked": len(sample)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
