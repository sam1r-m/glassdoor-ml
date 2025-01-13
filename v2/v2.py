# VADER and nltk (natural language toolkit)
# uses prebuilt sentiment analysis model that is typically used for neg/neu/pos classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

# Load the training and test datasets
df = pd.read_csv('small.csv')
#test_df = pd.read_csv('test.csv')

print(df.shape)

# Some EDA:
#ax = df['rating'].value_counts().sort_index().plot(kind='bar', title='Count of Ratings', figsize=(10, 5))
#ax.set_xlabel('Rating')
#plt.show()

example = df['cons'][19]
#print(example)

tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens) # tags each token with a set of predetermined codes from NLTK documentation

#print(tokens[0:10])
#print(tagged[0:10])

entities = nltk.chunk.ne_chunk(tagged) # chunks the words with their codes
# entities.pprint() # pretty printing !

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

sia = SentimentIntensityAnalyzer()

#print(sia.polarity_scores('I am so happy!'))
#print(sia.polarity_scores(example))

# We will now run the polarity_scores function on first 1000 rows
res = {}
for i, row in tqdm(df.iloc[:1000].iterrows(), total=1000):
    text = row['pros'] + row['cons']
    res[i] = sia.polarity_scores(text)

# Convert the results into a DataFrame
vaders = pd.DataFrame(res).T

# Reset index and merge with original DataFrame 'df'
vaders = vaders.reset_index().rename(columns={'index': 'i'})
vaders = vaders.merge(df, how='left', left_on='i', right_index=True)

# View the result
#print(vaders.head())

#ax = sns.barplot(data=vaders, x='compound', y='compound')
#ax.set_title('Distribution of Compound Sentiment Scores')
#plt.show()

#ax = sns.barplot(data=vaders, x='rating', y='compound')
#ax.set_title('Compound Sentiment Scores by Rating')
#plt.show()

""" fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='rating', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='rating', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='rating', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show() """

# Roberta Pretrained Model:
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#print(example)
print(sia.polarity_scores(example))

# Now we attempt to run the roBERTa model on it (based on Google's 2018 BERT model)
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

#print(polarity_scores_roberta(example))

# Iterate for all the scores:
res = {}
for i, row in tqdm(df.iloc[:1000].iterrows(), total=1000):
    try:
        text = row['pros'] + row['cons']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[i] = both
    except RuntimeError:
        print(f'Broke for index {i}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'i'})
results_df = results_df.merge(df, how='left', left_on='i', right_index=True)

print(results_df.columns)

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()
