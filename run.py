import yaml
import argparse
import numpy as np

from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

parser.add_argument('--load_model', 
                    action='store_true',
                    help='if loading an existing model',
                    )

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = vae_models[config['model_params']['name']](**config['model_params'])
model_filename = "model.pt"
if args.load_model:
    # model.load_state_dict(torch.load(model_filename))
model = model.double()
# model.eval()

hparams = argparse.Namespace(**config['exp_params'])
experiment = VAEXperiment(model,
                          config['exp_params'],
                          hparams)
# experiment = VAEXperiment.load_from_checkpoint("logs/VanillaVAE_12/version_114/checkpoints/_ckpt_epoch_29.ckpt")
# experiment.load_from_checkpoint("checkpoints/_ckpt_epoch_27.ckpt")

# checkpoint_callback = ModelCheckpoint(filepath='checkpoints')

runner = Trainer(default_save_path=f"{tt_logger.save_dir}",
                #  checkpoint_callback=checkpoint_callback,
                 min_nb_epochs=1,
                 logger=tt_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
torch.save(model.state_dict(), model_filename)

# runner.test(experiment)

print(next(model.parameters())[:10])
