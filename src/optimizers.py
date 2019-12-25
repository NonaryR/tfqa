from spanbert.optimization import get_linear_schedule_with_warmup, AdamW
from functools import partial


OPTS = {"adamw": partial(AdamW)}
SCHEDULERS = {"linear-warmup": get_linear_schedule_with_warmup}


def optimizer_params(model_named_parameters):
    param_optimizer = list(model_named_parameters)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    return optimizer_grouped_parameters


def get_optimizer(model_named_parameters, lr, optimizer_name="adamw"):
    optim = OPTS[optimizer_name]
    opt_params = optimizer_params(model_named_parameters) 

    return optim(opt_params, lr=lr, correct_bias=False)


def get_scheduler(optimizer, num_warmup_steps, num_train_optimization_steps, scheduler_name="linear-warmup"):
    scheduler = SCHEDULERS[scheduler_name](optimizer, num_warmup_steps, num_train_optimization_steps)
    return scheduler

