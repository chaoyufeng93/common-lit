from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from transfer_model import Hugging_Transfer
from train import train
from tqdm import tqdm
from data import Mydataset
from predict import predict



def stack(df, 
          df_ts, 
          n_fold = 5,
          tgt = 'content',
          fixed_text = False,
          batch_size = 8,
          mdl_li = [('/kaggle/input/deberta-v3-base/microsoft:deberta-v3-base', 'tok')],
          adapter_type = ['adapter_fixed_upper', 6],
          act_layer_num = 6,
          act_layer_method = 'down',
          act_layer_init = False,
          EPOCH = 10,
          early_stop = 4,
          loss_fuc = torch.nn.MSELoss(),
          lr = [8e-5, 8e-5, 3e-5]
          ):
    kfold = KFold(n_splits = n_fold)
    # ('/kaggle/input/roberta-base-go-emotions/SamLowe:roberta-base-go_emotions', 'not_tok')
    # ('/kaggle/input/bert-ner/dslim:bert-base-NER', 'tok')
    val_all = torch.Tensor()
    ts_all = torch.Tensor()

    for mdl_n in tqdm(range(len(mdl_li))):
        kfold = KFold(n_splits = 5)
        mdl_ = mdl_li[mdl_n][0]
        val_ = torch.Tensor()
        ts_ = torch.Tensor()

        for i, idx in enumerate(kfold.split(df)):
            # define dataset and dataloader
            tr_data = Mydataset(
                df.loc[idx[0], df.columns], 
                mdl_ = mdl_, 
                model_type = mdl_li[mdl_n][1], 
                target = tgt, 
                fixed_text = fixed_text
            )

            val_data = Mydataset(
                df.loc[idx[1], df.columns], 
                mdl_ = mdl_, 
                model_type = mdl_li[mdl_n][1], 
                target = tgt, 
                fixed_text = fixed_text
            )

            ts_data = Mydataset(
                df_ts[df_ts.columns], 
                mdl_=mdl_, 
                model_type = mdl_li[mdl_n][1], 
                mode = 'test', 
                target = tgt, 
                fixed_text = fixed_text
            )

            tr_dl = DataLoader(tr_data, batch_size = batch_size, shuffle = True, drop_last = False)
            val_dl = DataLoader(val_data, batch_size = batch_size)
            ts_dl = DataLoader(ts_data, batch_size = batch_size)
            print('-'*30 + 'Fold {}'.format(str(i+1)) + '-'*30)

            # define the model
            # adapter_upper -> [num: ]
            # adapter_down -> [:-num]
            model = Hugging_Transfer(mdl_ = mdl_, 
                                     tuning = adapter_type, 
                                     model_type = mdl_li[mdl_n][1])
            model.cuda()
            model.tuning_init()
            model.regressor_init()
            # :num[0] & -num[1]:
            if 'canine' in mdl_:
                model.activate_char()
            # :num[0] & -num[1]:
            model.activate_layer(
                num = act_layer_num, 
                method = act_layer_method, 
                init_params = act_layer_init
            )

             # training
            train(
                  model, 
                  tr_dl, 
                  val_dl, 
                  EPOCH = EPOCH,
                  early_stop_step = early_stop, 
                  lr = lr,
                  loss_f = loss_fuc,
                  save_kw = 'num{}_{}_{}'.format(mdl_n + 1, i + 1, tgt), 
                  model_type = mdl_li[mdl_n][1]
            )
            torch.cuda.empty_cache()

            best_mod = torch.load('mod_num{}_{}_{}.mdl'.format(mdl_n + 1, i + 1, tgt))
            model.load_state_dict(best_mod)

            # predict the fold (out)
            val_ = torch.cat([val_, predict(model, val_data, model_type = mdl_li[mdl_n][1])])
            ts_ = torch.cat([ts_, predict(model, ts_dl, model_type = mdl_li[mdl_n][1])], axis = 1)
            del model
            torch.cuda.empty_cache()
        ts_ = ts_.sum(axis = 1).reshape(-1, 1)
        
        val_all = torch.cat([val_all, val_], axis = 1)
        ts_all = torch.cat([ts_all, ts_], axis = 1)
    return val_all, ts_all
