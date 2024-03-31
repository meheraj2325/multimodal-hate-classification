import argparse

import pandas as pd
import pytorch_lightning as pl
import wandb
from config import (
    batch_size,
    early_stopping_patience,
    gpus,
    learning_rate,
    max_epochs,
    num_classes,
    train_backbone,
)

from datamodules import FBHateMemeDataModule
from models import GuidedAttentionModel


def train(args):
    # Initialize WandB
    wandb.init(
        project="Multimodal-Hate-Classification",
        entity="ccds-bangla-nlp",
        name="Training Guided Attention Model with freezed backbone",
        dir="experiments",
    )
    # Load dataset
    train_df = pd.read_csv(f"{args.data_dir}/{args.dataset}_train.csv")
    val_df = pd.read_csv(f"{args.data_dir}/{args.dataset}_val.csv")
    test_df = pd.read_csv(f"{args.data_dir}/{args.dataset}_test.csv")

    # Initialize data module
    data_module = FBHateMemeDataModule(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    # Initialize model
    fusion_model = GuidedAttentionModel(
        num_classes=num_classes,
        train_backbone=args.train_backbone,
        lr=args.learning_rate,
    )

    # Initialize callbacks
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.early_stopping_patience, mode="min"
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="experiments/checkpoints",
        filename="best_model",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        callbacks=[early_stopping_callback, model_checkpoint_callback],
        progress_bar_refresh_rate=20,
        logger=wandb,
    )

    # Train the model
    trainer.fit(fusion_model, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fusion Model")
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    parser.add_argument("--data_dir", type=str, help="Directory containing dataset")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument(
        "--batch_size", type=int, default=batch_size, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=max_epochs,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=early_stopping_patience,
        help="Patience for early stopping",
    )
    parser.add_argument("--gpus", type=int, default=gpus, help="Number of GPUs to use")
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        help="Set to train the parameters of both VIT and BERT models",
    )

    args = parser.parse_args()

    train(args)
