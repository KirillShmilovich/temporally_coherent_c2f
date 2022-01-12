from cvae_test import cVAE
import pytorch_lightning as pl
from pl_callbacks import CheckpointEveryNSteps
from argparse import Namespace


def main():
    pl.seed_everything(42)

    hparams = pl.core.saving.load_hparams_from_tags_csv("ADP_params.csv")
    # hparams["use_edm_bonds"] = True

    model = cVAE(**hparams)

    args = Namespace(**hparams)
    logger = pl.loggers.TensorBoardLogger("./", name="ADP_logs_final")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[CheckpointEveryNSteps(2000)],
        val_check_interval=0.25,
        gpus=1,
        terminate_on_nan=True,
        logger=logger,
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
