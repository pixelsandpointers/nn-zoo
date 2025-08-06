import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, utils
import wandb

from loss import vae_loss_function
from model import GeneralizedPixelVAE


# --- DDP Setup and Cleanup Functions ---
def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# --- Main Worker Function (for each GPU) ---
def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # W&B Initialization
    if rank == 0 and not args.no_wandb:
        run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=args,
        )

    # Data Loading
    transform = transforms.Compose([transforms.ToTensor()])
    if rank == 0:
        datasets.CIFAR10(root=args.data_path, train=True, download=True)
    dist.barrier()

    train_dataset = datasets.CIFAR10(
        root=args.data_path, train=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_path, train=False, transform=transform
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.vis_batch_size, shuffle=True
    )
    vis_images, _ = next(iter(test_loader))
    vis_images = vis_images.to(rank)

    # Model, Optimizer, and DDP Wrapper
    model = GeneralizedPixelVAE(input_dim=(3, 32, 32), latent_dim=args.latent_dim).to(
        rank
    )
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters(), lr=args.lr)

    if rank == 0 and not args.no_wandb:
        wandb.watch(ddp_model.module, log="all", log_freq=1000)

    # --- Training Loop ---
    global_step = 0
    # FIX: Calculate total steps for KL annealing schedule
    total_anneal_steps = args.kl_anneal_epochs * len(train_loader)

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()

        for i, (images, _) in enumerate(train_loader):
            images = images.to(rank)

            # FIX: Calculate current KL weight for annealing
            if total_anneal_steps > 0:
                kl_weight = min(1.0, global_step / total_anneal_steps)
            else:
                kl_weight = 1.0  # No annealing if epochs are 0

            logits, mu, log_var = ddp_model(images)
            # FIX: Pass the kl_weight to the loss function
            loss_dict = vae_loss_function(logits, images, mu, log_var, kl_weight)
            loss = loss_dict["total_loss"]

            optimizer.zero_grad()
            loss.backward()

            # FIX: Add gradient clipping for stability
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)

            optimizer.step()
            global_step += 1

            if rank == 0 and not args.no_wandb and (i + 1) % 100 == 0:
                log_data = {
                    "epoch": epoch,
                    "step": global_step,
                    "total_loss": loss_dict["total_loss"].item(),
                    "recon_loss": loss_dict["recon_loss"].item(),
                    "kld": loss_dict["kld"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "kl_weight": kl_weight,  # FIX: Log the kl_weight
                }
                wandb.log(
                    log_data
                )  # --- Visualization and Checkpointing at end of epoch ---
        if rank == 0:
            ddp_model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Get reconstructions for the fixed visualization batch
                recon_logits, _, _ = ddp_model.module(vis_images)

                # Convert logits to images
                recon_probs = F.softmax(
                    recon_logits.view(vis_images.shape[0], 3, 256, 32, 32), dim=2
                )
                recon_pixels = torch.argmax(recon_probs, dim=2)
                recon_images = recon_pixels.float() / 255.0

                # Combine original and reconstructed images for comparison
                comparison = torch.cat([vis_images.cpu(), recon_images.cpu()])
                grid = utils.make_grid(comparison, nrow=args.vis_batch_size)

                # W&B: Log the image grid
                if not args.no_wandb:
                    wandb.log(
                        {
                            "reconstructions": wandb.Image(
                                grid, caption=f"Epoch {epoch + 1}"
                            )
                        },
                        step=global_step,
                    )

            # Save model checkpoint
            if (epoch + 1) % args.save_interval == 0:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                checkpoint_path = os.path.join(
                    args.save_dir, f"model_epoch_{epoch + 1}.pth"
                )
                torch.save(ddp_model.module.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

    # W&B: Finish the run on the main process
    if rank == 0 and not args.no_wandb:
        wandb.finish()

    cleanup()


def main():
    parser = argparse.ArgumentParser(description="PixelVAE DDP")
    # Training args
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=64, type=int, help="batch size per GPU")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--latent-dim", default=128, type=int)
    parser.add_argument("--data-path", default="./data", type=str)
    parser.add_argument("--save-dir", default="./checkpoints", type=str)
    parser.add_argument("--save-interval", default=5, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument(
        "--vis-batch-size",
        default=8,
        type=int,
        help="batch size for visualization images",
    )

    # FIX: Add arguments for KL annealing and gradient clipping
    parser.add_argument(
        "--kl-anneal-epochs", default=10, type=int, help="epochs to anneal KL term over"
    )
    parser.add_argument(
        "--grad-clip",
        default=1.0,
        type=float,
        help="max norm for gradient clipping (0 to disable)",
    )

    # W&B: Add arguments for wandb control
    parser.add_argument(
        "--no-wandb", action="store_true", help="disable weights and biases logging"
    )
    parser.add_argument(
        "--wandb-project", default="nn-zoo", type=str, help="wandb project name"
    )
    parser.add_argument(
        "--wandb-entity",
        default="benjaminbeilharz",
        type=str,
        help="wandb entity (user or team name)",
    )

    args = parser.parse_args()

    world_size = 2
    mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
