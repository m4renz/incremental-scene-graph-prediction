from lightning import LightningDataModule, LightningModule
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import MLFlowLogger
from ssg_tools.dataset.hetero_dataset import (  # noqa F401
    FullSceneGraphModule,
    HeteroSceneGraphModule,
    HomoSceneGraphModule,
)
from ssg_tools.models.ksgn import (  # noqa F401
    IncrementalKSGN,
    IncrementalKSGNHomo,
    KSGNRandomClassifier,
    IncrementalKSGNLinear,
    IncrementalKSGNFull,
)

from ssg_tools.models.sgfn import SGFN  # noqa F401


class MLFlowSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage):
        if isinstance(trainer.logger, MLFlowLogger):
            run_id = trainer.logger.run_id
            trainer.logger.experiment.log_dict(run_id, self.config.as_dict(), "config.yaml")


def cli_main():
    cli = LightningCLI(  # noqa F841
        LightningModule,
        LightningDataModule,
        save_config_callback=MLFlowSaveConfigCallback,
        save_config_kwargs={"save_to_log_dir": False},
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    cli_main()
