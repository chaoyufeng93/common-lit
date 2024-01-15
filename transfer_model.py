import torch
from transformers import AutoModel

class Hugging_Transfer(torch.nn.Module):
    def __init__(
        self, 
        mdl_, 
        tuning = ['adapter_smooth', 1], 
        model_type = 'tok'
    ):
        super(Hugging_Transfer,self).__init__()
        self.mdl_ = mdl_
        self.tuning = tuning
        self.model_type = model_type

        self.model = AutoModel.from_pretrained(mdl_)

        for p in self.model.parameters():
            p.requires_grad = False


        if tuning[0] == 'adapter_smooth':

            for _, layer in enumerate(self.model.encoder.layer):
                if _ % tuning[1] == 0:
                    layer.attention.output.dense.add_module(
                        'adapter',
                    torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                    )
                )

                    for p in layer.attention.output.LayerNorm.parameters():
                        p.requires_grad = True

                    layer.output.dense.add_module(
                        'adapter',
                    torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                  )
                )

                    for p in layer.output.LayerNorm.parameters():
                        p.requires_grad = True

        # activate_layer upper -> [: num]
        elif tuning[0] == 'adapter_fixed_upper':

            for _, layer in enumerate(self.model.encoder.layer[tuning[1]:]):
                layer.attention.output.dense.add_module(
                        'adapter',
                        torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                    )
                )

                for p in layer.attention.output.LayerNorm.parameters():
                    p.requires_grad = True

                layer.output.dense.add_module(
                        'adapter',
                        torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                  )
                )

                for p in layer.output.LayerNorm.parameters():
                    p.requires_grad = True

        # activate_layer down -> [-num:]
        elif tuning[0] == 'adapter_fixed_down':

            for _, layer in enumerate(self.model.encoder.layer[:-tuning[1]]):
                layer.attention.output.dense.add_module(
                        'adapter',
                        torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                    )
                )

                for p in layer.attention.output.LayerNorm.parameters():
                    p.requires_grad = True

                layer.output.dense.add_module(
                        'adapter',
                        torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                  )
                )

                for p in layer.output.LayerNorm.parameters():
                    p.requires_grad = True

        elif tuning[0] == 'adapter_fixed_medium':

            for _, layer in enumerate(self.model.encoder.layer[tuning[1][0]: -tuning[1][1]]):
                layer.attention.output.dense.add_module(
                        'adapter',
                        torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                    )
                )

                for p in layer.attention.output.LayerNorm.parameters():
                    p.requires_grad = True

                layer.output.dense.add_module(
                        'adapter',
                        torch.nn.Sequential(
                        torch.nn.Linear(768, 256),
                        torch.nn.ReLU(),
                        # torch.nn.Dropout(p = 0.1),
                        torch.nn.Linear(256, 768)
                  )
                )

                for p in layer.output.LayerNorm.parameters():
                    p.requires_grad = True

        elif tuning[0] == 'lora':
            pass


        try:
            if self.model.pooler:
                for p in self.model.pooler.parameters():
                    p.requires_grad = True
        except:
            for p in self.model.encoder.rel_embeddings.parameters():
                p.requires_grad = True

            for p in self.model.encoder.LayerNorm.parameters():
                p.requires_grad = True

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 384),
            torch.nn.ReLU(),
            torch.nn.Linear(384, 1),
            torch.nn.Dropout(p = 0.1)
        )

    def tuning_init(self):
        if self.tuning[0] == 'adapter_smooth':

            for _, layer in enumerate(self.model.encoder.layer):
                if _ % self.tuning[1] == 0:
                    torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[0].weight)
                    torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[2].weight)

                    torch.nn.init.xavier_normal_(layer.output.dense.adapter[0].weight)
                    torch.nn.init.xavier_normal_(layer.output.dense.adapter[2].weight)

            print('adapter smooth parameters initialized!')

        elif self.tuning[0] == 'adapter_fixed_upper':

            for _, layer in enumerate(self.model.encoder.layer[self.tuning[1]:]):
                torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[0].weight)
                torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[2].weight)

                torch.nn.init.xavier_normal_(layer.output.dense.adapter[0].weight)
                torch.nn.init.xavier_normal_(layer.output.dense.adapter[2].weight)

            print('adapter fixed upper parameters initialized!')

        elif self.tuning[0] == 'adapter_fixed_down':

            for _, layer in enumerate(self.model.encoder.layer[:-self.tuning[1]]):
                torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[0].weight)
                torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[2].weight)

                torch.nn.init.xavier_normal_(layer.output.dense.adapter[0].weight)
                torch.nn.init.xavier_normal_(layer.output.dense.adapter[2].weight)

            print('adapter fixed down parameters initialized!')

        elif self.tuning[0] == 'adapter_fixed_medium':
            for _, layer in enumerate(self.model.encoder.layer[self.tuning[1][0]: -self.tuning[1][1]]):
                torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[0].weight)
                torch.nn.init.xavier_normal_(layer.attention.output.dense.adapter[2].weight)

                torch.nn.init.xavier_normal_(layer.output.dense.adapter[0].weight)
                torch.nn.init.xavier_normal_(layer.output.dense.adapter[2].weight)


            print('adapter fixed medium parameters initialized!')

        else:
            print('no such an adapter option')


    def regressor_init(self):

        torch.nn.init.xavier_normal_(self.regressor[0].weight)
        torch.nn.init.xavier_normal_(self.regressor[2].weight)
        print('regressor parameters initialized!')

    def activate_layer(self, num = 3, method = 'smooth', init_params = False):
        if method == 'smooth':
            for i, layer in enumerate(self.model.encoder.layer):
                if init_params == True:
                    for i in layer.parameters():
                        torch.nn.init.normal_(i)
                if i % num == 0:
                    for p in layer.parameters():
                        p.requires_grad = True

        # adapter_upper -> [num:]
        elif method == 'upper':
            for i, layer in enumerate(self.model.encoder.layer[:num]):
                if init_params == True:
                    for i in layer.parameters():
                        torch.nn.init.normal_(i)
                for p in layer.parameters():
                    p.requires_grad = True

        # adapter_down -> [:-num]
        elif method == 'down':
            for i, layer in enumerate(self.model.encoder.layer[-num:]):
                if init_params == True:
                    for i in layer.parameters():
                        torch.nn.init.normal_(i)
                for p in layer.parameters():
                    p.requires_grad = True

        elif method == 'both':
            for i, layer in enumerate(self.model.encoder.layer[:num[0]]):
                if init_params == True:
                    for i in layer.parameters():
                        torch.nn.init.normal_(i)
                for p in layer.parameters():
                    p.requires_grad = True

            for i, layer in enumerate(self.model.encoder.layer[-num[1]:]):
                if init_params == True:
                    for i in layer.parameters():
                        torch.nn.init.normal_(i)
                for p in layer.parameters():
                    p.requires_grad = True

        else:
            print('No such a method')

    def activate_char(self):
        if 'canine' in self.mdl_:
          for p in self.model.final_char_encoder.parameters():
              p.requires_grad = True
          print('char activated!')


    def forward(self, *args):
        if self.model_type == 'tok':
            ids, tok, att = args[0].view(args[0].shape[0], -1), args[1].view(args[1].shape[0], -1), args[2].view(args[2].shape[0], -1)

            x = self.model(input_ids = ids,
                           token_type_ids = tok,
                           attention_mask = att)

        else:
            ids, att = args[0].view(args[0].shape[0], -1), args[1].view(args[1].shape[0], -1)

            x = self.model(input_ids = ids, attention_mask = att)

        last = x.last_hidden_state.sum(axis = 1) / x.last_hidden_state.shape[1]
        out = self.regressor(last)

        return out