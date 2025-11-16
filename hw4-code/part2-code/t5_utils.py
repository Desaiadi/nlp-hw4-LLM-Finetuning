import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def setup_wandb(args):
    wandb.init(
        project="text-to-sql-t5",
        name=args.experiment_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": args.max_n_epochs,
            "finetune": args.finetune,
        },
    )


def initialize_model(args):
    if args.finetune:
        print("Loading pretrained T5-small...")
        model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    else:
        print("Initializing T5-small from scratch...")
        config = T5Config.from_pretrained("google-t5/t5-small")
        model = T5ForConditionalGeneration(config)

    model = model.to(DEVICE)
    print(
        f"Model on {DEVICE}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params"
    )
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best):
    mkdir(checkpoint_dir)
    save_path = os.path.join(
        checkpoint_dir, "best_model.pt" if best else "latest_model.pt"
    )
    if best:
        print(f"Saving best model to {save_path}")
    torch.save({"model_state_dict": model.state_dict()}, save_path)


def load_model_from_checkpoint(args, best):
    checkpoint_dir = args.checkpoint_dir
    load_path = os.path.join(
        checkpoint_dir, "best_model.pt" if best else "latest_model.pt"
    )
    print(f"Loading from {load_path}")

    model = initialize_model(args)
    checkpoint = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(
        model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS
    )
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999),
        )
    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    else:
        raise NotImplementedError


def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result
