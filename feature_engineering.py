from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import pyphen
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from collections import Counter
from nltk import ne_chunk, word_tokenize, pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer

import spacy
import re
from autocorrect import Speller
from spellchecker import SpellChecker
import lightgbm as lgb
tqdm.pandas()

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

dic = pyphen.Pyphen(lang='en')
sid = SentimentIntensityAnalyzer()

class Preprocessor:
    def __init__(self
                # model_name: str,
                ) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.twd = TreebankWordDetokenizer()
        self.STOP_WORDS = set(stopwords.words('english'))

        self.spacy_ner_model = spacy.load('en_core_web_sm',)
        self.speller = Speller(lang='en')
        self.spellchecker = SpellChecker()

    def calculate_text_similarity(self, row):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([row['prompt_text'], row['text']])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0]

    def sentiment_analysis(self, text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity, analysis.sentiment.subjectivity

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """
        def check_is_stop_word(word):
            return word in self.STOP_WORDS

        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))

    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)

    def ner_overlap_count(self, row, mode:str):
        model = self.spacy_ner_model
        def clean_ners(ner_list):
            return set([(ner[0].lower(), ner[1]) for ner in ner_list])
        prompt = model(row['prompt_text'])
        summary = model(row['text'])

        if "spacy" in str(model):
            prompt_ner = set([(token.text, token.label_) for token in prompt.ents])
            summary_ner = set([(token.text, token.label_) for token in summary.ents])
        elif "stanza" in str(model):
            prompt_ner = set([(token.text, token.type) for token in prompt.ents])
            summary_ner = set([(token.text, token.type) for token in summary.ents])
        else:
            raise Exception("Model not supported")

        prompt_ner = clean_ners(prompt_ner)
        summary_ner = clean_ners(summary_ner)

        intersecting_ners = prompt_ner.intersection(summary_ner)

        ner_dict = dict(Counter([ner[1] for ner in intersecting_ners]))

        if mode == "train":
            return ner_dict
        elif mode == "test":
            return {key: ner_dict.get(key) for key in self.ner_keys}


    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary)>0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):

        wordlist=text.split()
        amount_miss = len(list(self.spellchecker.unknown(wordlist)))

        return amount_miss

    def calculate_unique_words(self,text):
        unique_words = set(text.split())
        return len(unique_words)

    def add_spelling_dictionary(self, tokens: List[str]) -> List[str]:
        """dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens)
        self.speller.nlp_data.update({token:1000 for token in tokens})

    def calculate_pos_ratios(self , text):
        pos_tags = pos_tag(nltk.word_tokenize(text))
        pos_counts = Counter(tag for word, tag in pos_tags)
        total_words = len(pos_tags)
        ratios = {tag: count / total_words for tag, count in pos_counts.items()}
        return ratios

    def calculate_punctuation_ratios(self,text):
        total_chars = len(text)
        punctuation_counts = Counter(char for char in text if char in '.,!?;:"()[]{}')
        ratios = {char: count / total_chars for char, count in punctuation_counts.items()}
        return ratios

    def calculate_keyword_density(self,row):
        keywords = set(row['prompt_text'].split())
        text_words = row['text'].split()
        keyword_count = sum(1 for word in text_words if word in keywords)
        return keyword_count / len(text_words)

    def count_syllables(self,word):
        hyphenated_word = dic.inserted(word)
        return len(hyphenated_word.split('-'))

    def flesch_reading_ease_manual(self,text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        total_syllables = sum(self.count_syllables(word) for word in TextBlob(text).words)

        if total_sentences == 0 or total_words == 0:
            return 0

        flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        return flesch_score

    def flesch_kincaid_grade_level(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        total_syllables = sum(self.count_syllables(word) for word in TextBlob(text).words)

        if total_sentences == 0 or total_words == 0:
            return 0

        fk_grade = 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59
        return fk_grade

    def gunning_fog(self, text):
        total_sentences = len(TextBlob(text).sentences)
        total_words = len(TextBlob(text).words)
        complex_words = sum(1 for word in TextBlob(text).words if self.count_syllables(word) > 2)

        if total_sentences == 0 or total_words == 0:
            return 0

        fog_index = 0.4 * ((total_words / total_sentences) + 100 * (complex_words / total_words))
        return fog_index

    def calculate_sentiment_scores(self,text):
        sentiment_scores = sid.polarity_scores(text)
        return sentiment_scores

    def count_difficult_words(self, text, syllable_threshold=3):
        words = TextBlob(text).words
        difficult_words_count = sum(1 for word in words if self.count_syllables(word) >= syllable_threshold)
        return difficult_words_count



    def run(self,
            prompts: pd.DataFrame,
            summaries:pd.DataFrame,
            mode:str
        ) -> pd.DataFrame:

        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(word_tokenize(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: word_tokenize(x)
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(word_tokenize(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: word_tokenize(x)
        )
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(
            lambda x: self.speller(x)
        )
        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(
            lambda x: self.add_spelling_dictionary(x)
        )

        prompts['gunning_fog_prompt'] = prompts['prompt_text'].apply(self.gunning_fog)
        prompts['flesch_kincaid_grade_level_prompt'] = prompts['prompt_text'].apply(self.flesch_kincaid_grade_level)
        prompts['flesch_reading_ease_prompt'] = prompts['prompt_text'].apply(self.flesch_reading_ease_manual)


#         from IPython.core.debugger import Pdb; Pdb().set_trace()
        # fix misspelling
#         summaries["fixed_summary_text"] = summaries["text"].progress_apply(
#             lambda x: self.speller(x)
#         )


        # count misspelling
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")
        input_df['flesch_reading_ease'] = input_df['text'].apply(self.flesch_reading_ease_manual)
        input_df['word_count'] = input_df['text'].apply(lambda x: len(x.split()))
        input_df['sentence_length'] = input_df['text'].apply(lambda x: len(x.split('.')))
        input_df['vocabulary_richness'] = input_df['text'].apply(lambda x: len(set(x.split())))

        input_df['word_count2'] = [len(t.split(' ')) for t in input_df.text]
        input_df['num_unq_words']=[len(list(set(x.lower().split(' ')))) for x in input_df.text]
        input_df['num_chars']= [len(x) for x in input_df.text]

        # Additional features
        input_df['avg_word_length'] = input_df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
        input_df['comma_count'] = input_df['text'].apply(lambda x: x.count(','))
        input_df['semicolon_count'] = input_df['text'].apply(lambda x: x.count(';'))

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']

        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence,args=(2,), axis=1
        )
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)

        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)

        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)

        input_df['exclamation_count'] = input_df['text'].apply(lambda x: x.count('!'))
        input_df['question_count'] = input_df['text'].apply(lambda x: x.count('?'))
        input_df['pos_ratios'] = input_df['text'].apply(self.calculate_pos_ratios)

        # Convert the dictionary of POS ratios into a single value (mean)
        input_df['pos_mean'] = input_df['pos_ratios'].apply(lambda x: np.mean(list(x.values())))
        input_df['punctuation_ratios'] = input_df['text'].apply(self.calculate_punctuation_ratios)

        # Convert the dictionary of punctuation ratios into a single value (sum)
        input_df['punctuation_sum'] = input_df['punctuation_ratios'].apply(lambda x: np.sum(list(x.values())))
        input_df['keyword_density'] = input_df.apply(self.calculate_keyword_density, axis=1)
        input_df['jaccard_similarity'] = input_df.apply(lambda row: len(set(word_tokenize(row['prompt_text'])) & set(word_tokenize(row['text']))) / len(set(word_tokenize(row['prompt_text'])) | set(word_tokenize(row['text']))), axis=1)
        tqdm.pandas(desc="Performing Sentiment Analysis")
        input_df[['sentiment_polarity', 'sentiment_subjectivity']] = input_df['text'].progress_apply(
            lambda x: pd.Series(self.sentiment_analysis(x))
        )
        tqdm.pandas(desc="Calculating Text Similarity")
        input_df['text_similarity'] = input_df.progress_apply(self.calculate_text_similarity, axis=1)
        #Calculate sentiment scores for each row
        input_df['sentiment_scores'] = input_df['text'].apply(self.calculate_sentiment_scores)

        input_df['gunning_fog'] = input_df['text'].apply(self.gunning_fog)
        input_df['flesch_kincaid_grade_level'] = input_df['text'].apply(self.flesch_kincaid_grade_level)
        input_df['count_difficult_words'] = input_df['text'].apply(self.count_difficult_words)

        # Convert sentiment_scores into individual columns
        sentiment_columns = pd.DataFrame(list(input_df['sentiment_scores']))
        input_df = pd.concat([input_df, sentiment_columns], axis=1)
        input_df['sentiment_scores_prompt'] = input_df['prompt_text'].apply(self.calculate_sentiment_scores)
        # Convert sentiment_scores_prompt into individual columns
        sentiment_columns_prompt = pd.DataFrame(list(input_df['sentiment_scores_prompt']))
        sentiment_columns_prompt.columns = [col +'_prompt' for col in sentiment_columns_prompt.columns]
        input_df = pd.concat([input_df, sentiment_columns_prompt], axis=1)
        columns =  ['pos_ratios', 'sentiment_scores', 'punctuation_ratios', 'sentiment_scores_prompt']
        cols_to_drop = [col for col in columns if col in input_df.columns]
        if cols_to_drop:
            input_df = input_df.drop(columns=cols_to_drop)

        print(cols_to_drop)
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])

