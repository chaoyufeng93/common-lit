import torch
from tqdm import tqdm



def train(
        model, 
        tr_dl, 
        val_dl, 
        EPOCH = 5, 
        early_stop_step = 4, 
        lr = [5e-5, 5e-5, 3e-5], 
        loss_f = torch.nn.MSELoss(), 
        save_kw = str(1), 
        model_type = 'tok'
):
    loss_catch_tr = []
    loss_catch_val = []
    best_loss = 100
    stop_count = 0

    if torch.cuda.is_available():
        print("Model will be training on GPU")
        model.cuda()
        loss_f = loss_f.cuda()
    else:
        print("Model will be training on CPU")

  #opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay = 0.01)
    try:
        if model.model.pooler:
            opt = torch.optim.AdamW([
                {'params': model.model.encoder.parameters()},
                {'params': model.model.pooler.parameters(), 'lr': lr[0]},
                {'params': model.regressor.parameters(), 'lr': lr[1]}
            ], lr = lr[2], weight_decay = 0.01)
    except:
        opt = torch.optim.AdamW([
                {'params': model.model.encoder.layer.parameters()},
                {'params': model.model.encoder.rel_embeddings.parameters(), 'lr': lr[0]},
                {'params': model.model.encoder.LayerNorm.parameters(), 'lr': lr[0]},
                {'params': model.regressor.parameters(), 'lr': lr[1]}
            ], lr = lr[2], weight_decay = 0.01)

    for i in tqdm(range(EPOCH)):
        loss_c = 0
        loss_v = 0
        rmse_c = 0
        rmse_cv = 0
        model.train()
        for i, bag in enumerate(tr_dl):

            if model_type == 'tok':
                ids, tok, att, y = bag[0].cuda(), bag[1].cuda(), bag[2].cuda(), bag[3].cuda()
                y_pre = model(ids, tok, att)
            else:
                ids, att, y = bag[0].cuda(), bag[1].cuda(), bag[2].cuda()
                y_pre = model(ids, att)

            # content_loss, word_loss = torch.sqrt(loss_f(y_pre[:, 0], y[:, 0])), torch.sqrt(loss_f(y_pre[:, 1], y[:, 1]))
            # content_loss, word_loss = loss_f(y_pre[:, 0], y[:, 0]), loss_f(y_pre[:, 1], y[:, 1])
            loss = torch.sqrt(loss_f(y_pre, y))
            loss_rmse = loss

            loss_c += loss
            rmse_c += loss_rmse

            opt.zero_grad()
            loss.backward()
            opt.step()


        model.eval()
        with torch.no_grad():
            for i, bag in enumerate(val_dl):
                if model_type == 'tok':
                    ids, tok, att, y = bag[0].cuda(), bag[1].cuda(), bag[2].cuda(), bag[3].cuda()
                    y_pre = model(ids, tok, att)
                else:
                    ids, att, y = bag[0].cuda(), bag[1].cuda(), bag[2].cuda()
                    y_pre = model(ids, att)

                #content_loss, word_loss = torch.sqrt(loss_f(y_pre[:, 0], y[:, 0])), torch.sqrt(loss_f(y_pre[:, 1], y[:, 1]))
                # content_loss, word_loss = loss_f(y_pre[:, 0], y[:, 0]), loss_f(y_pre[:, 1], y[:, 1])

                loss = torch.sqrt(loss_f(y_pre, y))
                loss_rmse = loss
                loss_v += loss
                rmse_cv += loss_rmse

            if best_loss > loss_v/len(val_dl):
                stop_count = 0
                best_loss = loss_v/len(val_dl)
                torch.save(model.state_dict(), 'mod_{}.mdl'.format(save_kw))
                print('best saved')
            else:
                stop_count += 1

        print('train loss: {}'.format(loss_c/len(tr_dl)))
        print('train rmse loss: {}'.format(rmse_c/len(tr_dl)))
        print('val loss: {}'.format(loss_v/len(val_dl)))
        print('val rmse loss: {}'.format(rmse_cv/len(val_dl)))

        if stop_count >= early_stop_step:
            print('early stopping!')
            break

        loss_catch_tr.append(loss_c/len(tr_dl))
        loss_catch_val.append(loss_v/len(val_dl))
