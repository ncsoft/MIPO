import logging
import numpy as np
import torch
import transformers
import random


from utils.data import get_benchmark_datasets
from utils.configs import (ModelArguments, DataArguments, PEFTArguments)
from utils.configs import H4ArgumentParser
import os
from trl import DPOConfig
from trainers.mipo_trainer import MIPOTrainer


logger = logging.getLogger(__name__)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def train():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, PEFTArguments))
    model_args, data_args, training_args, peft_args = parser.parse()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    training_args.output_dir = f"{training_args.output_dir}/{data_args.tag}-mipo-beta{training_args.beta}-lr{training_args.learning_rate}"
    logger.info(f"OUTPUT_DIR: {training_args.output_dir}")

    # Set seed
    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.random.manual_seed(training_args.seed)
    torch.cuda.manual_seed(training_args.seed)
    torch.cuda.manual_seed_all(training_args.seed)

    ##################################################
    ## load datasets
    ##################################################
    raw_datasets = get_benchmark_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"]
    )

    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )


    logger.info(f"# of train data: {len(raw_datasets['train'])}")
    if "test" in raw_datasets:
        logger.info(f"# of test data: {len(raw_datasets['test'])}")

    ###############################################
    # Set tokenizer
    ###############################################

    if model_args.tokenizer_name_or_path:
        tokenizer_path = model_args.tokenizer_name_or_path
    else:
        tokenizer_path = model_args.model_name_or_path

    #####################################
    # Load tokenizer and process datasets
    #####################################

    # data_args.truncation_side = "left"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=True,
        legacy=False
    )

    raw_datasets = raw_datasets.remove_columns("messages")

    ######################
    # For Model
    ######################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    model = model_args.model_name_or_path
    ref_model = model
    ref_model_kwargs = model_kwargs


    #########################
    # Instantiate MIPO trainer
    #########################
    trainer = MIPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        model_init_kwargs=model_kwargs,
        padding_value=tokenizer.eos_token_id,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.args.do_eval = False

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    train()
