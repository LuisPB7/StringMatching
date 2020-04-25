import argparse
import glob
import pandas as pd
import pickle
import torch
from RSModel import RSModel
from RSPoolModel import RSPoolModel
from RSPenTanhModel import RSPenTanhModel
from RSPenTanhPoolModel import RSPenTanhPoolModel
from RSHardAttentionModel import RSHardAttentionModel
from TransformerModel import TransformerModel
from RTransformerModel import RTransformerModel, RMogrifierTransformerModel
from TransformerInteractionModel import TransformerInteractionModel
from RSPenTanhHardAttentionModel import RSPenTanhHardAttentionModel
from MogrifierLSTMModel import MogrifierLSTMModel
from SHARNNModel import SHARNNModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Define arguments to be passed by the user #
parser=argparse.ArgumentParser()
parser.add_argument('--model', help='What model')
parser.add_argument('--train', help='What dataset to train on')
parser.add_argument('--test', help='What dataset to test on')
args=parser.parse_args()
#############################################

models = {'rs':RSModel, 'rs_pool':RSPoolModel, 'rs_pentanh':RSPenTanhModel, 'rs_pentanh_pool':RSPenTanhPoolModel, 'rs_hardatt':RSHardAttentionModel, \
          'rs_pentanh_hardatt':RSPenTanhHardAttentionModel, 'transformer':TransformerModel, 'r_transformer': RTransformerModel, \
          'mogrifier_lstm': MogrifierLSTMModel, 'r_mogrifier_transformer': RMogrifierTransformerModel, 'transformer_interaction': TransformerInteractionModel, 'sha_rnn': SHARNNModel }

model = models[args.model](args.model, args.train, args.test)

early_stop_callback = EarlyStopping(
    monitor='loss',
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode='min'
)

trainer = Trainer(gpus=1, \
                  show_progress_bar=True, \
                  max_nb_epochs=20, \
                  early_stop_callback=early_stop_callback)

try:
    model.load_state_dict(torch.load('weights/{}-{}.pt'.format(args.model, args.train)),strict=True)
    print("Successfully loaded weights")
except:
    trainer.fit(model)

#trainer.fit(model)
trainer.test(model)
