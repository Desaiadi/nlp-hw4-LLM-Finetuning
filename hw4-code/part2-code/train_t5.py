import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PAD_IDX = 0


def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop")

    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--optimizer_type", type=str, default="AdamW")
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_epochs", type=int, default=0)
    parser.add_argument("--max_n_epochs", type=int, default=0)
    parser.add_argument("--patience_epochs", type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join(
        "checkpoints", f"{model_type}_experiments", args.experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # , change to:
    gt_sql_path = 'data/dev.sql'
    gt_record_path = 'records/ground_truth_dev.pkl'
    model_sql_path = f'results/t5_{model_type}_dev.sql'
    model_record_path = f'records/t5_{model_type}_dev.pkl'
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Train loss {tr_loss:.4f}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args,
            model,
            dev_loader,
            gt_sql_path,
            model_sql_path,
            gt_record_path,
            model_record_path,
        )
        print(
            f"Epoch {epoch}: Dev loss {eval_loss:.4f}, F1 {record_f1:.4f}, RecEM {record_em:.4f}, SQLEM {sql_em:.4f}, Err {error_rate:.2%}"
        )

        if args.use_wandb:
            wandb.log(
                {"train/loss": tr_loss, "dev/loss": eval_loss, "dev/f1": record_f1},
                step=epoch,
            )

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)

        if epochs_since_improvement >= args.patience_epochs:
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(
        train_loader, desc="Training"
    ):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )["logits"]

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(
    args,
    model,
    dev_loader,
    gt_sql_pth,
    model_sql_path,
    gt_record_path,
    model_record_path,
):
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    generated_queries = []

    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(
            dev_loader, desc="Evaluating"
        ):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            outputs = model.generate(input_ids=encoder_input, attention_mask=encoder_mask, max_length=512, num_beams=5, early_stopping=True)

            for output in outputs:
                decoded_query = tokenizer.decode(output, skip_special_tokens=True)
                generated_queries.append(decoded_query)

    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = len([msg for msg in model_error_msgs if msg != ""]) / len(
        model_error_msgs
    )

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    generated_queries = []
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Test"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            outputs = model.generate(input_ids=encoder_input, attention_mask=encoder_mask, max_length=512, num_beams=5, early_stopping=True)


            for output in outputs:
                decoded_query = tokenizer.decode(output, skip_special_tokens=True)
                generated_queries.append(decoded_query)

    print(f"Saving {len(generated_queries)} test predictions...")
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)


def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(
        args.batch_size, args.test_batch_size
    )
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(
        args, model, len(train_loader)
    )

    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    model = load_model_from_checkpoint(args, best=True)
    model.eval()

    model_type = "ft" if args.finetune else "scr"
    model_sql_path = f"results/t5_{model_type}_experiment_test.sql" #final-change-for autograder
    model_record_path = f"records/t5_{model_type}_experiment_test.pkl"  #final-change-for autograder
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print("\n" + "="*50)
    print("Training and inference complete!")
    print(f"Files saved:")
    print(f"  - {model_sql_path}")
    print(f"  - {model_record_path}")
    print("="*50)


if __name__ == "__main__":
    main()
