import argparse
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from open_musiclm.config import (
    load_model_config,
    load_training_config,
    create_hubert_kmeans_from_config,
    create_hubert_batch_kmeans_trainer_from_config,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train hubert kmeans using batch kmeans algorithm"
    )
    parser.add_argument("--results_folder", default=None, type=str)
    parser.add_argument("--project_dir", default=None, type=str)
    parser.add_argument("--model_config", default=None, type=str)
    parser.add_argument("--training_config", default=None, type=str)
    parser.add_argument("--continue_from", default=None, type=str)

    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    training_config = load_training_config(args.training_config)

    device = "cpu"

    print("loading hubert...")
    hubert_kmeans = create_hubert_kmeans_from_config(
        model_config, args.continue_from, device, use_batch_kmeans=True
    )
    print("creating trainer")
    trainer = create_hubert_batch_kmeans_trainer_from_config(
        model_config=model_config,
        training_config=training_config,
        hubert_kmeans=hubert_kmeans,
        results_folder=args.results_folder,
        accelerate_kwargs={"log_with": "tensorboard", "project_dir": args.project_dir},
        config_paths=[args.model_config, args.training_config],
    )
    trainer.train()
