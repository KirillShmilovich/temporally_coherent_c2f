from cvae_test import cVAE
import pytorch_lightning as pl
from pl_callbacks import CheckpointEveryNSteps
from argparse import Namespace


def main():
    pl.seed_everything(42)

    hparams = pl.core.saving.load_hparams_from_tags_csv("CLN_params.csv")
    # hparams["energy_loss_gradient_clip_val"] = 0.001
    # hparams["use_edm_bonds"] = True

    model = cVAE(**hparams)

    args = Namespace(**hparams)
    logger = pl.loggers.TensorBoardLogger("./", name="CLN_logs_final_v3")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[CheckpointEveryNSteps(2000)],
        val_check_interval=0.25,
        gpus=2,
        accelerator="ddp",
        terminate_on_nan=True,
        logger=logger,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
