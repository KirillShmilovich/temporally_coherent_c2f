"""
Example template for defining a system.
"""
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule

from data import C2FDataSet, C2FDataSetCLN
import mdtraj as md
import pathlib
from modules import Encoder, Decoder
from openmmBridge import OpenMMEnergy
from utils import avg_blob, make_grid_np, voxel_gauss, to_distmat
import numpy as np
from tqdm import tqdm


class cVAE(LightningModule):
    def __init__(
        self,
        path,
        aa_traj,
        aa_pdb,
        cg_traj,
        cg_pdb,
        n_atoms_aa,
        n_atoms_cg,
        sigma,
        resolution,
        length,
        n_frames,
        num_workers,
        batch_size,
        learning_rate,
        latent_dim,
        fac_encoder,
        fac_decoder,
        train_percent,
        E_mu,
        E_std,
        save_every_n_steps,
        hallucinate_every_n_epochs,
        use_edm_loss,
        use_coord_loss,
        use_cg_loss,
        bonds_edm_weight,
        cg_coord_weight,
        coord_weight,
        beta,
        learning_gamma,
        default_save_path,
        mol_name,
        tops_fname,
        energy_loss_after_n_steps,
        energy_loss_gradient_clip_val,
        energy_weight_start,
        energy_weight_end,
        energy_weight_anneal_steps,
        energy_loss_clamp,
        # use_edm_bonds,
        **kwargs,
    ):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()
        delta_s = self.hparams.length / self.hparams.resolution
        self.grid = make_grid_np(delta_s, self.hparams.resolution)
        self.grid_shape = (self.hparams.resolution,) * 3
        self.build_model()

    def build_model(self):

        n_channels_encoder = (
            self.hparams.n_frames
        ) * self.hparams.n_atoms_aa + self.hparams.n_atoms_cg
        n_channels_decoder = n_channels_encoder - self.hparams.n_atoms_aa

        self.encoder = Encoder(
            self.hparams.resolution,
            self.hparams.resolution,
            self.hparams.resolution,
            in_channels=n_channels_encoder,
            latent_dim=self.hparams.latent_dim,
            fac=self.hparams.fac_encoder,
            device=self.device,
        )

        self.decoder = Decoder(
            z_dim=self.hparams.latent_dim,
            condition_n_channels=n_channels_decoder,
            fac=self.hparams.fac_decoder,
            out_channels=self.hparams.n_atoms_aa,
            resolution=self.hparams.resolution,
            device=self.device,
        )

        aa_pdb_fname = str(
            pathlib.Path(self.hparams.path).joinpath(self.hparams.aa_pdb)
        )
        if self.hparams.mol_name == "ADP":
            self.ELoss = OpenMMEnergy(
                self.hparams.n_atoms_aa * 3,
                aa_pdb_fname,
                mol_name=self.hparams.mol_name,
            )
        elif self.hparams.mol_name == "CLN":
            self.ELoss = OpenMMEnergy(
                self.hparams.n_atoms_aa * 3,
                aa_pdb_fname,
                mol_name=self.hparams.mol_name,
                tops_fname=self.hparams.tops_fname,
            )

    @staticmethod
    def _mean_center(x):
        return x - x.mean(axis=1, keepdims=True)

    def _avg_blob(self, vox):
        return avg_blob(
            vox,
            res=self.hparams.resolution,
            width=self.hparams.length,
            sigma=self.hparams.sigma,
            device=self.device,
        )

    def encode(self, input):
        mu, logvar = self.encoder(input)
        return (mu, logvar)

    def decode(self, z, condition):
        result = self.decoder(z, condition)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def cg_func(self, aa):
        return aa[:, self.cg_idxs, ...]

    def forward(self, encoder_input, condition):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, condition)
        return (output, mu, logvar, z)

    def _process_batch(self, aa, aa_vox, cg, cg_vox):
        aa_vox_past = aa_vox[:, :-1].squeeze(1)
        aa_vox_current = aa_vox[:, -1]
        cg_vox_current = cg_vox[:, -1]
        cg_vox_past = cg_vox[:, :-1].squeeze(1)

        aa_past = aa[:, :-1].squeeze(1)
        aa_current = aa[:, -1]
        cg_current = cg[:, -1]
        cg_past = cg[:, :-1].squeeze(1)

        return (
            aa_past,
            aa_current,
            aa_vox_past,
            aa_vox_current,
            cg_past,
            cg_current,
            cg_vox_past,
            cg_vox_current,
        )

    def reconstruct_forward(self, aa_vox_past, aa_vox_current, cg_vox_current):
        condition = torch.cat((cg_vox_current, aa_vox_past), dim=1)

        encoder_input = torch.cat((condition, aa_vox_current), dim=1)
        (recon_aa_vox, mu, logvar, z) = self.forward(encoder_input, condition)

        aa_fake = self._avg_blob(recon_aa_vox)
        return recon_aa_vox, aa_fake, mu, logvar, z

    def loss_hallucinate(self, batch):
        bs = batch[0].size(0)

        (
            aa_past,
            aa_current,
            aa_vox_past,
            aa_vox_current,
            cg_past,
            cg_current,
            cg_vox_past,
            cg_vox_current,
        ) = self._process_batch(*batch)

        (recon_aa_vox, aa_fake, mu, logvar, z) = self.reconstruct_forward(
            aa_vox_past, aa_vox_current, cg_vox_current
        )

        voxel_loss = F.mse_loss(recon_aa_vox, aa_vox_current)
        coords_loss = F.mse_loss(self._mean_center(aa_fake), aa_current)
        recon_loss = voxel_loss + coords_loss

        fake_cg_coords = self.cg_func(aa_fake)
        cg_coord_loss = F.mse_loss(
            self._mean_center(fake_cg_coords),
            cg_current,
        )

        fake_edm = to_distmat(aa_fake)
        real_edm = to_distmat(aa_current)
        edm_loss = F.mse_loss(fake_edm, real_edm)

        real_bonds_edm = real_edm[:, self.bond_idxs[:, 0], self.bond_idxs[:, 1]]
        fake_bonds_edm = fake_edm[:, self.bond_idxs[:, 0], self.bond_idxs[:, 1]]
        bonds_edm_loss = F.mse_loss(fake_bonds_edm, real_bonds_edm)

        KLD_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0
        )
        KLD_weight = mu.size(0) / len(self.ds_train)

        recon_energy_loss_weight = self._get_recon_energy_weight()

        beta_weight = self._get_beta()
        loss = voxel_loss + beta_weight * KLD_weight * KLD_loss

        if self.use_recon_energy_loss:
            energies = (
                self.ELoss.energy(aa_current.reshape(bs, -1)) - self.hparams.E_mu
            ) / self.hparams.E_std

            recon_energies = (
                self.ELoss.energy(aa_fake.reshape(bs, -1)) - self.hparams.E_mu
            ) / self.hparams.E_std
            recon_energy_loss = torch.nn.functional.mse_loss(
                recon_energies, energies, reduction="none"
            )

            if self.hparams.energy_loss_clamp > 0:
                recon_energy_loss = recon_energy_loss.clamp_max(
                    self.hparams.energy_loss_clamp
                ).mean()

            recon_energy_loss = recon_energy_loss.mean()
            loss = loss + recon_energy_loss_weight * recon_energy_loss
        else:
            recon_energy_loss = torch.tensor([0.0], device=self.device)

        if self.hparams.use_edm_loss:
            # if self.hparams.use_edm_bonds:
            #    loss = loss + self.hparams.bonds_edm_weight * bonds_edm_loss / 2.0
            # else:
            loss = loss + self.hparams.bonds_edm_weight * edm_loss / 2.0
        if self.hparams.use_coord_loss:
            loss = loss + self.hparams.coord_weight * (coords_loss)
        if self.hparams.use_cg_loss:
            loss = loss + self.hparams.cg_coord_weight * cg_coord_loss

        loss_dict = {
            "loss": loss,
            "KLD": KLD_loss,
            "recon": recon_loss,
            "VOX": voxel_loss,
            "COORD": coords_loss,
            "Energy_recon": recon_energy_loss,
            "EDM": edm_loss,
            "bonds_EDM": bonds_edm_loss,
            "CG_coord": cg_coord_loss,
            "recon_energy_loss_weight": torch.tensor(
                recon_energy_loss_weight, device=self.device
            ),
            "kld_beta_weight": torch.tensor(beta_weight, device=self.device),
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        loss_dict = self.loss_hallucinate(batch)
        loss = loss_dict["loss"]
        tensorboard_logs = {("train/" + key): val for key, val in loss_dict.items()}

        return {"loss": loss, "log": tensorboard_logs}

    def on_train_epoch_start(self):
        print(f"Optimizer state after epoch {self.current_epoch}...")
        print(self.optimizers())

    def on_train_epoch_end(self, _):
        if self.current_epoch == 0:
            return
        if self.current_epoch % self.hparams.hallucinate_every_n_epochs == 0:
            test_loader = self.test_dataloader(shuffle=False)
            fake_coords = list()
            real_coords = list()

            with torch.no_grad():
                for batch_idx, batch in tqdm(
                    enumerate(test_loader),
                    total=len(test_loader),
                    leave=False,
                    desc="Hallucinating",
                ):
                    batch = self.transfer_batch_to_device(batch, self.device)
                    if batch_idx == 0:
                        (
                            aa_past,
                            aa_current,
                            aa_vox_past,
                            aa_vox_current,
                            cg_past,
                            cg_current,
                            cg_vox_past,
                            cg_vox_current,
                        ) = self._process_batch(*batch)
                    else:
                        (
                            _,
                            aa_current,
                            _,
                            _,
                            _,
                            _,
                            _,
                            cg_vox_current,
                        ) = self._process_batch(*batch)

                    (recon_aa_vox, aa_fake, mu, logvar, z) = self.reconstruct_forward(
                        aa_vox_past, aa_vox_current, cg_vox_current
                    )
                    aa_fake = self._mean_center(aa_fake)
                    aa_vox_recon = voxel_gauss(
                        aa_fake,
                        res=self.hparams.resolution,
                        width=self.hparams.length,
                        sigma=self.hparams.sigma,
                        device=self.device,
                    )
                    aa_vox_current = aa_vox_recon
                    aa_vox_past = torch.cat((aa_vox_past, aa_vox_recon), dim=1)[
                        :, self.hparams.n_atoms_aa :, ...
                    ]
                    fake_coords.append(aa_fake)
                    real_coords.append(aa_current)

                hallucinate_coords = torch.cat(fake_coords, dim=0)
                real_coords = torch.cat(real_coords, dim=0)

                hallucinate_trj = md.Trajectory(
                    hallucinate_coords.detach().cpu().numpy(), topology=self.aa_traj.top
                )
                real_trj = md.Trajectory(
                    real_coords.detach().cpu().numpy(), topology=self.aa_traj.top
                )
                samples_path = (
                    pathlib.Path(self.logger.save_dir)
                    .joinpath(self.logger.name)
                    .joinpath(f"version_{self.logger.version}")
                    .joinpath("samples")
                )
                samples_path.mkdir(parents=True, exist_ok=True)

                print(f"Saving hallucination on step {self.global_step}...")
                hallucinate_trj.save_pdb(str(samples_path / "hallucination.pdb"))
                real_trj.save_pdb(str(samples_path / "hallucination_real.pdb"))

    def on_train_batch_end(self, _, batch, batch_idx, dataloader_idx):
        if self.global_step % self.hparams.save_every_n_steps == 0:
            batch = next(iter(self.val_dataloader(shuffle=True)))
            batch = self.transfer_batch_to_device(batch, self.device)
            with torch.no_grad():
                (
                    aa_past,
                    aa_current,
                    aa_vox_past,
                    aa_vox_current,
                    cg_past,
                    cg_current,
                    cg_vox_past,
                    cg_vox_current,
                ) = self._process_batch(*batch)

                (recon_aa_vox, aa_fake, mu, logvar, z) = self.reconstruct_forward(
                    aa_vox_past, aa_vox_current, cg_vox_current
                )

                real_trj = md.Trajectory(
                    np.array(aa_fake.detach().cpu()), topology=self.aa_traj.top
                )
                fake_trj = md.Trajectory(
                    np.array(aa_current.detach().cpu()), topology=self.aa_traj.top
                )

                condition = torch.cat(
                    (
                        cg_vox_current,
                        aa_vox_past,
                    ),
                    dim=1,
                )

                repeats = 100
                z = torch.empty(
                    [repeats, self.hparams.latent_dim],
                    dtype=condition.dtype,
                    device=self.device,
                ).normal_()
                c = torch.repeat_interleave(
                    condition[-1].unsqueeze(0), repeats=repeats, dim=0
                )
                fake = self.decoder(z, c)
                fake_coords = self._avg_blob(fake)

                condition_coords = torch.repeat_interleave(
                    aa_past[-1].unsqueeze(0),
                    repeats=repeats,
                    dim=0,
                )

                z_trj = md.Trajectory(
                    np.array(fake_coords.detach().cpu()), topology=self.aa_traj.top
                )
                condition_trj = md.Trajectory(
                    np.array(condition_coords.detach().cpu()), topology=self.aa_traj.top
                )

                samples_path = (
                    pathlib.Path(self.logger.save_dir)
                    .joinpath(self.logger.name)
                    .joinpath(f"version_{self.logger.version}")
                    .joinpath("samples")
                )
                samples_path.mkdir(parents=True, exist_ok=True)

                print(f"Saving samples on step {self.global_step}")
                real_trj.save_pdb(str(samples_path / "real_step.pdb"))
                fake_trj.save_pdb(str(samples_path / "fake_step.pdb"))
                z_trj.save_pdb(str(samples_path / "z_samples.pdb"))
                condition_trj.save_pdb(str(samples_path / "condition.pdb"))

    def _get_recon_energy_weight(self):
        if self.global_step >= self.hparams.energy_loss_after_n_steps:
            self.use_recon_energy_loss = True

            self.trainer.gradient_clip_val = self.hparams.energy_loss_gradient_clip_val

            factor = (
                self.hparams.energy_weight_end / self.hparams.energy_weight_start
            ) ** (1 / self.hparams.energy_weight_anneal_steps)

            recon_energy_loss_weight = self.hparams.energy_weight_start * (
                factor ** (self.global_step - self.hparams.energy_loss_after_n_steps)
            )
            recon_energy_loss_weight = min(
                self.hparams.energy_weight_end, recon_energy_loss_weight
            )
            return recon_energy_loss_weight
        self.use_recon_energy_loss = False
        return 0.0

    def _get_beta(self):
        if isinstance(self.hparams.beta, float):
            beta_weight = self.hparams.beta
        else:
            if isinstance(self.hparams.beta, str):
                beta_params = self.hparams.beta.split(",")
            elif isinstance(self.hparams.beta, tuple):
                beta_params = list(self.hparams.beta)

            if len(beta_params) == 4:  # Sigmoid Cyclic KL-beta
                k, x0, M, R = [float(f) for f in beta_params]
                if self.global_step < (M * R):
                    tau = np.mod(self.global_step, R)
                    beta_weight = float(1.0 / (1.0 + np.exp(-k * (tau - x0))))
                else:
                    beta_weight = 1.0

            if len(beta_params) == 3:  # Linear cliclic KL-beta
                T, M, R = [float(f) for f in beta_params]
                if self.global_step < (M * R):
                    tau = np.mod(self.global_step, R)
                    beta_weight = min(tau / T, 1.0)
                else:
                    beta_weight = 1.0

            if len(beta_params) == 2:  # Sigmoid anneal
                k, x0 = beta_params
                beta_weight = float(
                    1.0 / (1.0 + np.exp(-float(k) * (self.global_step - float(x0))))
                )
            if len(beta_params) == 1:  # Linear anneal
                R = float(beta_params[0])
                beta_weight = min(self.global_step / R, 1.0)
        return beta_weight

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        loss_dict = self.loss_hallucinate(batch)
        tensorboard_logs = {("val/" + key): val for key, val in loss_dict.items()}
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        keys = outputs[0].keys()
        tensorboard_logs = {
            key: torch.stack([x[key] for x in outputs]).mean() for key in keys
        }
        avg_loss = tensorboard_logs["val/loss"]
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def estimate_latent_space_density(self, n_components=10):

        dataloader = self.train_dataloader()
        latent_codes = list()
        Energies = list()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=False,
                desc="Estimating latent space density",
            ):
                batch = self.transfer_batch_to_device(batch, self.device)
                (
                    aa_past,
                    aa_current,
                    aa_vox_past,
                    aa_vox_current,
                    cg_past,
                    cg_current,
                    cg_vox_past,
                    cg_vox_current,
                ) = self._process_batch(*batch)
                bs = aa_past.size(0)
                energies = self.ELoss.energy(aa_current.reshape(bs, -1))
                (recon_aa_vox, aa_fake, mu, logvar, z) = self.reconstruct_forward(
                    aa_vox_past, aa_vox_current, cg_vox_current
                )

                latent_codes.append(z.detach().cpu().numpy())
                Energies.append(energies.detach().cpu().numpy())

        from sklearn.mixture import GaussianMixture

        z = np.concatenate(latent_codes)
        Energies = np.concatenate(Energies)

        self.GMM = GaussianMixture(n_components=n_components)
        print("Fitting GMM to posterior latent space density...")
        self.GMM.fit(z)
        return z, Energies

    def sample_GMM_latent_space(self, n_samples=1):
        z_GMM, _ = self.GMM.sample(n_samples)
        z_GMM = torch.Tensor(z_GMM).to(self.device)
        return z_GMM

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.learning_gamma
        )
        return [optimizer], [scheduler]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if self.hparams.mol_name == "ADP":
            self.aa_traj = md.load(
                str(pathlib.Path(self.hparams.path).joinpath(self.hparams.aa_traj)),
                top=str(pathlib.Path(self.hparams.path).joinpath(self.hparams.aa_pdb)),
            ).center_coordinates()
            self.cg_traj = md.load(
                str(pathlib.Path(self.hparams.path).joinpath(self.hparams.cg_traj)),
                top=str(pathlib.Path(self.hparams.path).joinpath(self.hparams.cg_pdb)),
            ).center_coordinates()
            self.cg_idxs = [4, 6, 8, 10, 14, 16]
        elif self.hparams.mol_name == "CLN":
            self.aa_traj = md.load(
                str(pathlib.Path(self.hparams.path).joinpath(self.hparams.aa_pdb))
            )
            self.cg_traj = md.load(
                str(pathlib.Path(self.hparams.path).joinpath(self.hparams.cg_pdb)),
            )
            trj_fnames = [
                str(f)
                for f in sorted(
                    pathlib.Path(self.hparams.path).joinpath("AA_xtc").glob("*xtc")
                )
            ]
            self.aa_trajs = [
                md.load(trj_fname, top=self.aa_traj.top).center_coordinates()
                for trj_fname in trj_fnames
            ]
            trj_fnames = [
                str(f)
                for f in sorted(
                    pathlib.Path(self.hparams.path).joinpath("CG_xtc").glob("*xtc")
                )
            ]
            self.cg_trajs = [
                md.load(trj_fname, top=self.cg_traj.top).center_coordinates()
                for trj_fname in trj_fnames
            ]
            self.cg_idxs = self.aa_traj.top.select("name CA")

        self.bond_idxs = torch.tensor(
            [[b.atom1.index, b.atom2.index] for b in self.aa_traj.top.bonds],
            device=self.device,
        ).long()

        print(f"Atoms in AA: {[a.name for a in self.aa_traj.topology.atoms]}")

        if self.hparams.mol_name == "ADP":
            print(
                f"AA trajectory with {self.aa_traj.n_frames} frames, {self.aa_traj.n_atoms} atoms"
            )
            print(
                f"CG trajectory with {self.cg_traj.n_frames} frames, {self.cg_traj.n_atoms} atoms"
            )
            n_train = int(self.aa_traj.n_frames * self.hparams.train_percent)
            print(
                f"Using {n_train} frames for training, {self.aa_traj.n_frames - n_train} for validation."
            )
            aa_coords_train = self.aa_traj.xyz[:n_train]
            cg_coords_train = self.cg_traj.xyz[:n_train]
            aa_coords_val = self.aa_traj.xyz[n_train:]
            cg_coords_val = self.cg_traj.xyz[n_train:]

            self.ds_train = C2FDataSet(
                coords_aa=aa_coords_train,
                coords_cg=cg_coords_train,
                sigma=self.hparams.sigma,
                resolution=self.hparams.resolution,
                length=self.hparams.length,
                rand_rot=True,
                n_frames=self.hparams.n_frames,
            )

            self.ds_val = C2FDataSet(
                coords_aa=aa_coords_val,
                coords_cg=cg_coords_val,
                sigma=self.hparams.sigma,
                resolution=self.hparams.resolution,
                length=self.hparams.length,
                rand_rot=False,
                n_frames=self.hparams.n_frames,
            )

        elif self.hparams.mol_name == "CLN":
            n_train = int(len(self.aa_trajs) * self.hparams.train_percent)
            print(
                f"Using {n_train} trjs for training, {len(self.aa_trajs) - n_train} for validation."
            )
            aa_coords_train = [trj.xyz for trj in self.aa_trajs[:n_train]]
            cg_coords_train = [trj.xyz for trj in self.cg_trajs[:n_train]]
            aa_coords_val = [trj.xyz for trj in self.aa_trajs[n_train:]]
            cg_coords_val = [trj.xyz for trj in self.cg_trajs[n_train:]]

            self.ds_train = C2FDataSetCLN(
                coords_aa=aa_coords_train,
                coords_cg=cg_coords_train,
                sigma=self.hparams.sigma,
                resolution=self.hparams.resolution,
                length=self.hparams.length,
                rand_rot=True,
                n_frames=self.hparams.n_frames,
            )

            self.ds_val = C2FDataSetCLN(
                coords_aa=aa_coords_val,
                coords_cg=cg_coords_val,
                sigma=self.hparams.sigma,
                resolution=self.hparams.resolution,
                length=self.hparams.length,
                rand_rot=False,
                n_frames=self.hparams.n_frames,
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self, shuffle=False):
        return DataLoader(
            self.ds_val,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self, shuffle=False):
        return DataLoader(
            self.ds_val,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_model_specific_args(parser, root_dir):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        # parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument(
            "--path",
            default="/project2/andrewferguson/Kirill/midway3_c2f/ADP_data",
            type=str,
        )
        parser.add_argument("--aa_traj", default="AA.dcd", type=str)
        parser.add_argument("--aa_pdb", default="AA.pdb", type=str)
        parser.add_argument("--cg_traj", default="CG.dcd", type=str)
        parser.add_argument("--cg_pdb", default="CG.pdb", type=str)
        parser.add_argument("--n_atoms_aa", default=22, type=int)
        parser.add_argument("--n_atoms_cg", default=6, type=int)

        parser.add_argument("--sigma", default=0.01, type=float)
        parser.add_argument("--resolution", default=12, type=float)
        parser.add_argument("--length", default=1.8, type=float)

        parser.add_argument("--n_frames", default=2, type=int)

        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)

        parser.add_argument("--latent_dim", default=32, type=int)
        parser.add_argument("--fac_encoder", default=8, type=int)
        parser.add_argument("--fac_decoder", default=8, type=int)

        parser.add_argument("--train_percent", default=0.95, type=float)

        parser.add_argument("--E_mu", default=-7.3970, type=float)
        parser.add_argument("--E_std", default=5.7602, type=float)

        parser.add_argument("--save_every_n_steps", default=6000, type=int)
        parser.add_argument("--hallucinate_every_n_epochs", default=5, type=int)

        parser.add_argument("--use_edm_loss", default=True, type=bool)
        parser.add_argument("--use_coord_loss", default=True, type=bool)
        parser.add_argument("--use_cg_loss", default=True, type=bool)
        parser.add_argument("--bonds_edm_weight", default=0.1, type=float)
        parser.add_argument("--cg_coord_weight", default=0.1, type=float)
        parser.add_argument("--coord_weight", default=1.0, type=float)
        parser.add_argument("--default_save_path", default=None)
        parser.add_argument("--beta", default=1.0, type=float)
        parser.add_argument("--learning_gamma", default=1.0, type=float)

        parser.add_argument("--mol_name", default="ADP", type=str)
        parser.add_argument(
            "--tops_fname",
            default="/project2/andrewferguson/Kirill/midway3_c2f/charmm_tops",
            type=str,
        )
        parser.add_argument(
            "--energy_loss_after_n_steps",
            default=200000,
            type=int,
        )
        parser.add_argument(
            "--energy_loss_gradient_clip_val",
            default=0.1,
            type=float,
        )
        parser.add_argument(
            "--energy_weight_start",
            default=1e-6,
            type=float,
        )
        parser.add_argument(
            "--energy_weight_end",
            default=1.0,
            type=float,
        )
        parser.add_argument(
            "--energy_weight_anneal_steps",
            default=1250000,
            type=int,
        )
        parser.add_argument(
            "--energy_loss_clamp",
            default=float,
            type=0.0,
        )

        # parser.add_argument("--use_edm_bonds", default=False, type=bool)

        return parser
