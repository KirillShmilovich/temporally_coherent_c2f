from cvae_test import cVAE
import pytorch_lightning as pl
from pl_callbacks import CheckpointEveryNSteps
from argparse import Namespace


def main():
    pl.seed_everything(42)

    # hparams = pl.core.saving.load_hparams_from_tags_csv("CLN_params.csv")
    # hparams["energy_loss_gradient_clip_val"] = 0.001

    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_v1/version_0/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_v1/R3/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs/R3/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_BONDS_v1/R2/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_BONDS/R2/checkpoints/N-Step-Checkpoint.ckpt"

    # hparams_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final/version_0/hparams.yaml"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final/version_0/checkpoints/N-Step-Checkpoint.ckpt"

    # hparams_path = (
    #    "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final/R1/hparams.yaml"
    # )
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final/R1/checkpoints/N-Step-Checkpoint.ckpt"
    # hparams = pl.core.saving.load_hparams_from_yaml(hparams_path)

    # hparams_path = (
    #    "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final/R3/hparams.yaml"
    # )
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final/R3/checkpoints/N-Step-Checkpoint.ckpt"

    # hparams_path = (
    #    "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs/R5/hparams.yaml"
    # )
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs/R5/checkpoints/N-Step-Checkpoint.ckpt"

    hparams_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final_v3/version_0/hparams.yaml"
    ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/CLN_logs_final_v3/version_0/checkpoints/N-Step-Checkpoint.ckpt"

    hparams = pl.core.saving.load_hparams_from_yaml(hparams_path)

    model = cVAE.load_from_checkpoint(ckpt_path, **hparams)

    args = Namespace(**hparams)
    logger = pl.loggers.TensorBoardLogger("./", name="CLN_logs_final_v3", version="R1")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[CheckpointEveryNSteps(2000)],
        val_check_interval=0.25,
        gpus=2,
        accelerator="ddp",
        terminate_on_nan=True,
        logger=logger,
        resume_from_checkpoint=ckpt_path,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
