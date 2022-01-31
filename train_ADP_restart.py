from cvae_test import cVAE
import pytorch_lightning as pl
from pl_callbacks import CheckpointEveryNSteps
from argparse import Namespace


def main():
    pl.seed_everything(42)

    # hparams = pl.core.saving.load_hparams_from_tags_csv("ADP_params.csv")

    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs/version_0/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs/version_1/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs/version_2/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_bonds/version_0/checkpoints/N-Step-Checkpoint.ckpt"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_bonds/version_1/checkpoints/N-Step-Checkpoint.ckpt"

    # hparams_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_final/version_0/hparams.yaml"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_final/version_0/checkpoints/N-Step-Checkpoint.ckpt"

    # hparams_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_final/version_1/hparams.yaml"
    # ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_final/version_1/checkpoints/N-Step-Checkpoint.ckpt"

    hparams_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_final/version_2/hparams.yaml"
    ckpt_path = "/project2/andrewferguson/Kirill/c2f_vae_final/ADP_logs_final/version_2/checkpoints/N-Step-Checkpoint.ckpt"

    hparams = pl.core.saving.load_hparams_from_yaml(hparams_path)
    model = cVAE.load_from_checkpoint(ckpt_path, **hparams)

    args = Namespace(**hparams)
    logger = pl.loggers.TensorBoardLogger("./", name="ADP_logs_final")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[CheckpointEveryNSteps(2000)],
        val_check_interval=0.25,
        gpus=1,
        terminate_on_nan=True,
        logger=logger,
        resume_from_checkpoint=ckpt_path,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
