from data import pre_merge
from stack import stack
import torch
import argparse
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path', type = str, default = '/data/', help = 'the path of raw data'
    )

    parser.add_argument(
        '--n_fold', type = int, default = 5, help = 'number of fold'
    )

    parser.add_argument(
        '--model_li_c', type = list, default = [('microsoft:deberta-v3-base', 'tok')], help = 'type of model and the output type of content model'
    )

    parser.add_argument(
        '--model_li_w', type = list, default = [('dslim:bert-base-NER', 'tok')], help = 'type of model and the output type of wording model'
    )

    parser.add_argument(
        '--adpater_type', type = list, default = ['adapter_fixed_upper', 6], help = 'adapter method'
    )

    parser.add_argument(
        'act_layer_num', type = int, default = 6, help = 'number of layers for activation'
    )
        
    parser.add_argument(
        '--act_layer_method', type = str, default = 'down', help = 'activate layer method'
    )

    parser.add_argument(
        '--act_layer_init', type = bool, default = False, help = 'initialize the parameters of activated layers'
    )

    parser.add_argument(
        '--EPOCH', type = int, default = 5, help = 'number of training epoch'
    )

    parser.add_argument(
        '--early_stop', type = int, default = 4, help = 'early stop step'
    )

    parser.add_argument(
        '--lr', type = list, default = [8e-5, 8e-5, 3e-5], help = 'learning rate for hierarchy layers'
    )

    parser.add_argument(
        '--loss_fuc', type = int, default = 'MSE', help = 'type of loss function for dp model'
    )

    parser.add_argument(
        '--pred_method', type = str, default = 'avg', help = 'final predict method' 
    )

    args = parser.parse_args()

    # type of loss function
    loss_dict = {
        'MSE': torch.nn.MSELoss()
    }
    # load  &  merge data
    df, df_ts = pre_merge(args.path)

    # run stack method, output -> meta_learning_train_data, meta_predict_test_data
    val_all_c, ts_all_c = stack(df, 
                                df_ts, 
                                n_fold = args.n_fold,
                                tgt = 'content',
                                mdl_li = args.model_li_c,
                                adapter_type = args.adapter_type,
                                act_layer_num = args.act_layer_num,
                                act_layer_method = args.act_layer_method,
                                act_layer_init = args.act_layer_init,
                                EPOCH = args.EPOCH,
                                early_stop = args.early_stop,
                                loss_fuc = loss_dict[args.loss_fuc],
                                lr = args.lr
                            )
    
    val_all_w, ts_all_w = stack(df, 
                                df_ts, 
                                n_fold = args.n_fold,
                                tgt = 'wording',
                                mdl_li = args.model_li_w,
                                adapter_type = args.adapter_type,
                                act_layer_num = args.act_layer_num,
                                act_layer_method = args.act_layer_method,
                                act_layer_init = args.act_layer_init,
                                EPOCH = args.EPOCH,
                                early_stop = args.early_stop,
                                loss_fuc = loss_dict[args.loss_fuc],
                                lr = args.lr
                            )
    
    if args.pred_method == 'avg':
        # ts_all = ts_all/5
        # content_pred = ts_all[:, 0]
        # wording_pred = ts_all[:, 1]
        pass

    elif args.pred_method == 'lgbm':
        # feature_generate()
        # go_tuning()
        # learning & pred
        pass

    else:
        pass