import os
import yaml
import json
import glob
import itertools

import torch
import torch.nn as nn

from models import *
from models.g2p import symbols


def build_config(config_path: str):

    return {
        "audio": yaml.load(open(os.path.join(config_path, "preprocessing_config.yaml"), "r"), Loader=yaml.FullLoader),
        "models": yaml.load(open(os.path.join(config_path, "model_config.yaml"), "r"), Loader=yaml.FullLoader),
        "train": yaml.load(open(os.path.join(config_path, "train_config.yaml"), "r"), Loader=yaml.FullLoader)
    }


def load_acoustic(n_speakers: int, config: dict, stats: dict) -> nn.Module:
    # model initialization
    if config["models"]["fastspeech2"]["variance"]["duration_modelling"]["learn_alignment"] is False:
        model = SupervisedFastSpeech2(
            n_symbols=len(symbols),
            n_speakers=n_speakers,
            n_channels=config["audio"]["mel"]["channels"],
            config=config["models"]["fastspeech2"],
            stats=stats
        )
    else:
        model = UnsupervisedFastSpeech2(
            n_symbols=len(symbols),
            n_speakers=n_speakers,
            n_channels=config["audio"]["mel"]["channels"],
            config=config["models"]["fastspeech2"],
            stats=stats
        )

    return model


def load_vocoder(config: dict, checkpoint_path: str, use_complex: bool=False) -> nn.Module:
    # model initialization
    if use_complex:
        model = iSTFT(config=config["models"]["istft"])
    else:
        model = HifiGan(config=config["models"]["hifigan"])
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])

    return model


def initialize_acoustic(n_speakers: int, config: dict, checkpoint_path: str=None, stats: dict=None, device: torch.device=None) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    # model initialization
    model = load_acoustic(n_speakers, config, stats).to(device)

    # loss initialization
    if config["models"]["fastspeech2"]["variance"]["duration_modelling"]["learn_alignment"] is True:
        criterion = UnsupervisedFastSpeech2Loss(
            config=config["train"]["fastspeech2"]["loss"],
            pitch_feature_level=config["models"]["fastspeech2"]["variance"]["variance_embedding"]["pitch_feature"], 
            energy_feature_level=config["models"]["fastspeech2"]["variance"]["variance_embedding"]["energy_feature"],
            use_uv=config["models"]["fastspeech2"]["variance"]["variance_embedding"]["use_uv"]
        )
        d_model = config["models"]["fastspeech2"]["encoder_hidden"]
    else:
        criterion = SupervisedFastSpeech2Loss(
            pitch_feature_level=config["models"]["fastspeech2"]["variance"]["variance_embedding"]["pitch_feature"], 
            energy_feature_level=config["models"]["fastspeech2"]["variance"]["variance_embedding"]["energy_feature"],
            use_uv=config["models"]["fastspeech2"]["variance"]["variance_embedding"]["use_uv"]
        )
        d_model = config["models"]["fastspeech2"]["encoder_hidden"]

    optimizer = ScheduledOptim(
        model,
        optimize_config=config["train"]["fastspeech2"]["optimizer"],
        encoder_hidden=d_model,
        current_step=0
    )

    if checkpoint_path is not None:
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict["state_dict"])
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    return model, criterion, optimizer


def initialize_vocoder(config: dict, use_complex: bool=False, checkpoint_path: str=None, device: torch.device=None) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    # model initialization
    if use_complex:
        g = iSTFT(config=config["models"]["istft"]).to(device)
    else:
        g = HifiGan(config=config["models"]["hifigan"]).to(device)
    msd = MultiScaleDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    # optimizer initialization
    g_optim = torch.optim.AdamW(
        g.parameters(),
        config["train"]["hifigan"]["optimizer"]["learning_rate"], 
        betas=config["train"]["hifigan"]["optimizer"]["betas"],
        eps=config["train"]["hifigan"]["optimizer"]["eps"], 
        weight_decay=config["train"]["hifigan"]["optimizer"]["weight_decay"]
    )
    d_optim = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        config["train"]["hifigan"]["optimizer"]["learning_rate"], 
        betas=config["train"]["hifigan"]["optimizer"]["betas"],
        eps=config["train"]["hifigan"]["optimizer"]["eps"], 
        weight_decay=config["train"]["hifigan"]["optimizer"]["weight_decay"]
    )

    if checkpoint_path is not None:
        checkpoint_gen = scan_checkpoint(checkpoint_path, "g_")
        print(f"Load generator checkpoint from {checkpoint_gen}...")
        checkpoint_dis = scan_checkpoint(checkpoint_path, "do_")
        print(f"Load discriminator checkpoint from {checkpoint_dis}...")

        if checkpoint_gen is None or checkpoint_dis is None:
            print(f"Cant\"t find {'generator' if checkpoint_gen is None else 'discriminator'} checkpoint...")
        else:
            checkpoint_dict = torch.load(checkpoint_gen, map_location=device)
            g.load_state_dict(checkpoint_dict["generator"])

            # load weight
            checkpoint_dict = torch.load(checkpoint_dis, map_location=device)
            mpd.load_state_dict(checkpoint_dict["mpd"])
            msd.load_state_dict(checkpoint_dict["msd"])
            # load optimize
            g_optim.load_state_dict(checkpoint_dict["optim_g"])
            d_optim.load_state_dict(checkpoint_dict["optim_d"])

    return (g, msd, mpd), (g_optim, d_optim)


def save_information(output_path, m_config, d_speakers, d_stats):
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf8") as f:
        yaml.dump(m_config, f, default_flow_style=False)

    with open(os.path.join(output_path, "speakers.json"), "w", encoding="utf8") as f:
        json.dump(d_speakers, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_path, "stats.json"), "w", encoding="utf8") as f:
        json.dump(d_stats, f, ensure_ascii=False, indent=4)


def save_checkpoint(model, optimizer, filepath):
    print(f"Saving model and optimizer state to {filepath}")
    if isinstance(model, list):
        generator, mpd, msd = model
        optG, optD, _, _ = optimizer
        torch.save(
            {
                "state_dict": {
                    "generator": generator.state_dict(),
                    "mpd": mpd.state_dict(),
                    "msd": msd.state_dict()
                },
                "optimizer": {
                    "optG": optG.state_dict(),
                    "optD": optD.state_dict()
                }
            }, 
            filepath
        )

    else:
        torch.save({"state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()}, filepath)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
