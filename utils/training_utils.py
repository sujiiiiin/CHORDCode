from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type="cosine"):
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer, warm_up_steps, param_update_steps
        )
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer, warm_up_steps, param_update_steps
        )
    if scheduler_type == "constant":
        return get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    raise ValueError(f"Invalid scheduler type: {scheduler_type}")
