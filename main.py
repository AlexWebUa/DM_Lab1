import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Use english stopwords
stop = stopwords.words('english')

# Use English stemmer.
stemmer = SnowballStemmer('english')

raw_sms = pd.read_csv('sms-spam-corpus.csv', encoding='ISO-8859-1', engine='python')

# Strings without special characters, numbers and converted to lowercase
formatted_sms = raw_sms.v2.fillna('').astype(str)\
    .str.replace(r'[^A-Za-z ]', '', regex=True)\
    .replace('', np.nan, regex=False)\
    .str.lower()

# Put every stop-word into regex pattern
pattern = r'\b(?:{})\b'.format('|'.join(stop))

# Strings without stopwords
cleaned_formatted_sms = formatted_sms.str.replace(pattern, '', regex=True).str.replace(r'\s+', ' ', regex=True)


# Stemming
splitted = cleaned_formatted_sms.str.split().fillna('')
stemmed_sms = splitted.apply(lambda x: [stemmer.stem(y) for y in x]).str.join(' ')

