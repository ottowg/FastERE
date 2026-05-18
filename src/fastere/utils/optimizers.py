from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from fastere.utils.cprint import cp

no_decay_param = ["bias", "LayerNorm.weight", "layer_norm.weight"]


def get_optimizer(theta, config):
    """Configure AdamW optimizer with optional cosine LR schedule."""
    model_lr = config.get("model_lr", config.lr)
    decoder_lr = config.get("decoder_lr", config.lr)
    filter_lr = config.lr * config.filter_lr
    ner_lr = config.lr * config.ner_lr
    rel_lr = config.lr * config.rel_lr

    added_list = []
    optimizer_group_parameters = []
    if config.use_ner_prompt:
        optimizer_group_parameters.extend(
            get_params(theta, name="ner_prefix_encoder", lr=ner_lr * 10)
        )
        added_list.append("ner_prefix_encoder")

    if config.use_rel_prompt:
        optimizer_group_parameters.extend(
            get_params(theta, name="rel_prefix_encoder", lr=rel_lr * 10)
        )
        added_list.append("rel_prefix_encoder")

    optimizer_group_parameters.extend(get_params(theta, name="plm_model", lr=model_lr))
    added_list.append("plm_model")

    optimizer_group_parameters.extend(get_params(theta, name="filter", lr=filter_lr))
    added_list.append("filter")

    optimizer_group_parameters.extend(get_params(theta, name="ner_model", lr=ner_lr))
    added_list.append("ner_model")

    optimizer_group_parameters.extend(get_params(theta, name="rel_model", lr=rel_lr))
    added_list.append("rel_model")

    optimizer_group_parameters.extend(
        get_params(theta, name=None, lr=decoder_lr, added_list=added_list)
    )

    optimizer = AdamW(optimizer_group_parameters, lr=config.lr, eps=1e-8)

    if config.warmup:
        num_training_steps = calc_num_training_steps(theta)
        theta.num_training_steps = num_training_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_training_steps * config.warmup,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    return optimizer


def get_params(model, name, lr, exclude=[], added_list=[]):
    global no_decay_param

    def _filter(full_name, params, with_decay):
        return (
            (not name or name in full_name)
            and not any(p in full_name for p in exclude)
            and not any(p in full_name for p in added_list)
            and params.requires_grad
            and (with_decay ^ any(p in full_name for p in no_decay_param))
        )

    if model.config.debug:
        params_info = ""
        for n, p in model.named_parameters():
            if _filter(n, p, True) and ".encoder.layer" not in n:
                params_info += f"\n- {n}, {list(p.size())}"
        if params_info:
            print(f"\nParameters (except encoder.layer) LR: {lr:.2e}{params_info}")

    with_decay = [p for n, p in model.named_parameters() if _filter(n, p, True)]
    without_decay = [p for n, p in model.named_parameters() if _filter(n, p, False)]
    optimizer_group = [
        {"params": with_decay, "lr": lr, "weight_decay": 0.01},
        {"params": without_decay, "lr": lr, "weight_decay": 0},
    ]

    if optimizer_group[0]["params"] == []:
        if name:
            print(cp.yellow(f"[WARNING] {name} could not be added to optimizer group."))
        return []
    return optimizer_group


def calc_num_training_steps(theta):
    trainer = theta.trainer
    if (
        isinstance(trainer.limit_train_batches, int)
        and trainer.limit_train_batches != 0
    ):
        dataset_size = trainer.limit_train_batches
    elif isinstance(trainer.limit_train_batches, float):
        dataset_size = len(trainer.datamodule.train_dataloader())
        dataset_size = int(dataset_size * trainer.limit_train_batches)
    else:
        dataset_size = len(trainer.datamodule.train_dataloader())

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = trainer.accumulate_grad_batches * num_devices
    max_estimated_steps = (dataset_size // effective_batch_size) * trainer.max_epochs

    if trainer.max_steps > 0 and trainer.max_steps < max_estimated_steps:
        return trainer.max_steps
    return max_estimated_steps
