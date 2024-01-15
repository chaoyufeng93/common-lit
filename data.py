import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def pre_merge(path):
    samp_sub = pd.read_csv(path + 'sample_submission.csv')
    promp_tr = pd.read_csv(path + 'prompts_train.csv')
    summ_ts = pd.read_csv(path + 'summaries_test.csv')
    summ_tr = pd.read_csv(path + 'summaries_train.csv')
    promp_ts = pd.read_csv(path + 'prompts_test.csv')

    df = pd.merge(summ_tr, promp_tr, how = 'left', on = ['prompt_id'])
    df_ts = pd.merge(summ_ts, promp_ts, how = 'left', on = ['prompt_id'])

    df['text'], df['prompt_title'], df['prompt_question'] = df['text'].apply(lambda x: x.lower()), df['prompt_title'].apply(lambda x: x.lower()), df['prompt_question'].apply(lambda x: x.lower())
    df_ts['text'], df_ts['prompt_title'], df_ts['prompt_question'] = df_ts['text'].apply(lambda x: x.lower()), df_ts['prompt_title'].apply(lambda x: x.lower()), df_ts['prompt_question'].apply(lambda x: x.lower())
    df['prompt_text'], df_ts['prompt_text'] = df['prompt_text'].apply(lambda x: x.lower()), df_ts['prompt_text'].apply(lambda x: x.lower())
    return df, df_ts

class Mydataset(Dataset):
    def __init__(
            self, 
            df, 
            mdl_, 
            mode = None, 
            model_type = 'tok',
            target = 'content',
            fixed_text = False
    ): # model_type : [encoder, decoder]
        super(Mydataset,self).__init__()
        self.token = AutoTokenizer.from_pretrained(mdl_)
        self.t, self.q = list(df.prompt_title), list(df.prompt_question)
        self.x = list(df.text) if fixed_text == False else list(df.fixed_summary_texts)
        self.mode = mode
        self.model_type = model_type
        if mode != 'test':
            self.y = df[[target]].values


    def __getitem__(self,idx):
        x = self.token(
            self.t[idx] + self.token.sep_token + self.q[idx] + self.token.sep_token + self.x[idx],
            truncation=True,
            padding='max_length',
            max_length = 512,
            return_tensors = 'pt'
                      )


        ids, att = x['input_ids'], x['attention_mask']
        tok = x['token_type_ids'] if self.model_type == 'tok' else None

        if self.mode != 'test':
            y = self.y[idx]
            if tok != None:
                return ids, tok, att, torch.from_numpy(y).float()
            else:
                return ids, att, torch.from_numpy(y).float()

        else:
            if tok != None:
                return ids, tok, att
            else:
                return ids, att

    def __len__(self):

        return len(self.x)