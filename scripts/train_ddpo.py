
import argparse
import os
import torch
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mattergen.rl.ddpo import DDPOConfig, DDPOTrainer, DDPOSampler
from mattergen.common.utils.eval_utils import load_model_diffusion, MatterGenCheckpointInfo
from mattergen.common.utils.globals import get_device
from mattergen.common.data.datamodule import CrystDataModule
from mattergen.property_predictors import (
    BulkModulusLoRAMLPTimePredictor,
    BulkModulusLoRATimePredictor,
    BulkModulusTimeClassifier,
)

def _load_reward_model(ckpt_path):
    print(f"Loading reward model from {ckpt_path}")
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Inspect the checkpoint to decide which model class to use
    # Logic copied from generator.py
    predictor_search = ckpt.get("config", {}).get("args", {})
    if isinstance(predictor_search, dict):
        predictor_type = predictor_search.get("predictor_type", "mlp")
    else:
        predictor_type = "mlp"

    if predictor_type == "lora":
        return BulkModulusLoRATimePredictor.from_checkpoint(ckpt_path, device=device)
    if predictor_type == "lora_mlp":
        return BulkModulusLoRAMLPTimePredictor.from_checkpoint(ckpt_path, device=device)

    # Original logic for standard models
    model_cfg = ckpt.get("config", {}).get("model_kwargs", {})
    model = (
        BulkModulusTimeClassifier(**model_cfg)
        if isinstance(model_cfg, dict) and model_cfg
        else BulkModulusTimeClassifier()
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train DDPO for MatterGen")
    
    # Paths
    parser.add_argument("--diffusion_ckpt", default="checkpoints/mattergen_base", help="Path or name of diffusion checkpoint")
    parser.add_argument("--reward_ckpt", default="checkpoints/bulk_modulus_classifier_mixed_16_large/best.pt", help="Path to reward model checkpoint")
    parser.add_argument("--data_root", default="datasets/cache/mp_20", help="Root directory for cached dataset")
    parser.add_argument("--output_dir", default="outputs/ddpo_checkpoints", help="Directory to save checkpoints")
    
    # RL Hyperparameters
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_batches_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--actor_lr", type=float, default=1e-5)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of diffusion steps (N) for sampling")
    parser.add_argument("--train_epochs", type=int, default=10, help="Number of outer training loops")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Load Diffusion Model
    print(f"Loading diffusion model from {args.diffusion_ckpt}")
    if os.path.exists(args.diffusion_ckpt):
         ckpt_info = MatterGenCheckpointInfo(model_path=Path(args.diffusion_ckpt).resolve())
    else:
         ckpt_info = MatterGenCheckpointInfo.from_hf_hub(args.diffusion_ckpt)
         
    diffusion_module = load_model_diffusion(ckpt_info).to(device)
    
    # 2. Load Reward/Critic Model
    reward_model = _load_reward_model(args.reward_ckpt).to(device)
    
    # 3. Setup Config
    config = DDPOConfig(
        ppo_epochs=args.ppo_epochs,
        num_batches_per_epoch=args.num_batches_per_epoch,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        num_inference_steps=args.num_inference_steps
    )
    
    # 4. Initialize Trainer
    trainer = DDPOTrainer(diffusion_module, reward_model, config, device)
    
    # 5. Initialize Sampler
    # Use the corruption from the loaded model
    # diffusion_module here is DiffusionLightningModule, so we access the inner module
    sampler = DDPOSampler(
        corruption=diffusion_module.diffusion_module.corruption,
        score_fn=diffusion_module.diffusion_module.score_fn,
        device=device,
        N=config.num_inference_steps,
    )
    
    # 6. Setup Data Module
    print(f"Setting up data module with root: {args.data_root}")

    # We manually instantiate CrystDataModule to avoid Hydra complexity
    # We use minimal defaults that match mp_20.yaml structure where possible
    
    # Transforms (Partial instantiation usually, but here we can just use the classes if simple)
    # Actually, CrystDataModule expects config-like dicts or hydra instantiation for datasets.
    
    # Let's try to construct the dict config for instantiation
    # Mirroring mp_20.yaml logic but simplified
    
    # We need to resolve absolute path for data_root
    abs_data_root = os.path.abspath(args.data_root)
    
    # Constructing a config dictionary that `instantiate` can digest
    # or just instantiating CrystDataModule directly if we know the signature.
    # Signature: 
    # def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, num_workers, ...):
    
    # It takes dataset configs usually.
    # Let's import the Dataset class directly to pass instances or partials?
    # `CrystDataModule` instantiates datasets internally if passed as config.
    
    from mattergen.common.data.dataset import CrystalDataset
    
    # Define transforms
    # In mp_20.yaml: symmetrize_lattice, set_chemical_system_string
    # set_chemical_system_string is crucial for some models.
    from mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
    from mattergen.common.data.dataset_transform import filter_sparse_properties
    from functools import partial
    
    # Transforms list
    transforms = [
        partial(symmetrize_lattice),
        partial(set_chemical_system_string)
    ]
    
    dataset_transforms = [
        partial(filter_sparse_properties)
    ]
    
    # We need to define the dataset factory
    # The datamodule calls `instantiate(self.train_dataset_conf)`
    
    # Instantiate Datasets directly
    train_dataset = None
    if os.path.exists(os.path.join(abs_data_root, "train")):
        train_dataset = CrystalDataset.from_cache_path(
            cache_path=os.path.join(abs_data_root, "train"),
            transforms=transforms,
            dataset_transforms=dataset_transforms
        )
    
    val_dataset = None
    if os.path.exists(os.path.join(abs_data_root, "val")):
        val_dataset = CrystalDataset.from_cache_path(
            cache_path=os.path.join(abs_data_root, "val"),
            transforms=transforms,
            dataset_transforms=dataset_transforms
        )

    test_dataset = None
    if os.path.exists(os.path.join(abs_data_root, "test")):
        test_dataset = CrystalDataset.from_cache_path(
            cache_path=os.path.join(abs_data_root, "test"),
            transforms=transforms,
            dataset_transforms=dataset_transforms
        )
    
    if train_dataset is None:
        raise FileNotFoundError(f"Train dataset not found in {abs_data_root}/train")

    # Create simple config objects for DataModule
    pass_config_train = OmegaConf.create({"train": 0, "val": 0, "test": 0})
    pass_config_batch = OmegaConf.create({"train": int(args.batch_size), "val": int(args.batch_size), "test": int(args.batch_size)})

    # Instantiate DataModule
    datamodule = CrystDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=pass_config_batch,
        num_workers=pass_config_train
    )
    
    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()
    
    # 7. Prepare output directory with config.yaml so mattergen-generate can load checkpoints.
    # load_from_checkpoint_and_config expects {"state_dict": ...} + config.yaml alongside.
    import shutil
    os.makedirs(args.output_dir, exist_ok=True)

    base_ckpt_dir = Path(args.diffusion_ckpt).resolve() if os.path.exists(args.diffusion_ckpt) else None
    if base_ckpt_dir is not None:
        src_config = base_ckpt_dir / "config.yaml"
        dst_config = Path(args.output_dir) / "config.yaml"
        if src_config.exists() and not dst_config.exists():
            shutil.copy2(src_config, dst_config)

    print("Starting Training Loop...")

    # 8. Run Training — intermediate checkpoints are saved to output_dir during training.
    # Use --model_path=<output_dir> with mattergen-generate after training.
    trainer.train_loop(
        sampler=sampler,
        dataloader=train_dataloader,
        num_epochs=args.train_epochs,
        save_path=args.output_dir,
        save_every=args.save_every,
    )

    # Save final checkpoint (train_loop already saves last.ckpt each epoch, this is a safety copy).
    final_path = os.path.join(args.output_dir, "last.ckpt")
    torch.save({"state_dict": diffusion_module.state_dict()}, final_path)
    print(f"Done! Generate with: mattergen-generate <output_dir> --model_path={args.output_dir} ...")

if __name__ == "__main__":
    main()
