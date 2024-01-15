import torch

def predict(model, dataloader, model_type = 'tok'):
    model.eval()
    pred = torch.Tensor()

    with torch.no_grad():
        for i, bag in enumerate(dataloader):
            if model_type == 'tok':
                ids, tok, att = bag[0].cuda(), bag[1].cuda(), bag[2].cuda()
                y_pre = model(ids, tok, att)

            else:
                ids, att = bag[0].cuda(), bag[1].cuda()
                y_pre = model(ids, att)

            pred = torch.cat([pred, y_pre.cpu()])

    return pred