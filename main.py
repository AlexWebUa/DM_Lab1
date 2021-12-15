import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from matplotlib import pyplot as plt

# Use english stopwords
stop = stopwords.words('english')

# Use English stemmer.
stemmer = SnowballStemmer('english')

# 1. Reading from file
print('1. Reading from file')
raw_sms = pd.read_csv('sms-spam-corpus.csv', encoding='ISO-8859-1', engine='python')
dataframe = pd.DataFrame({'type': raw_sms.v1, 'message': raw_sms.v2})

# 2.1, 2.2 Deleting special characters and numbers; converting to lowercase
print('2.1, 2.2 Deleting special characters and numbers; converting to lowercase')
formatted_sms = dataframe.message.fillna('').astype(str) \
    .str.replace(r'[^A-Za-z ]', '', regex=True) \
    .replace('', np.nan, regex=False) \
    .str.lower()
dataframe['formatted'] = formatted_sms

# 2.3 Deleting stopwords
print('2.3 Deleting stopwords')
stopwords_pattern = r'\b(?:{})\b'.format('|'.join(stop))
cleaned_formatted_sms = formatted_sms.str \
    .replace(stopwords_pattern, '', regex=True).str \
    .replace(r'\s+', ' ', regex=True)
dataframe['cleaned'] = cleaned_formatted_sms

# 2.4 Stemming
print('2.4 Stemming')
splitted = cleaned_formatted_sms.str.split().fillna('')
stemmed_sms = splitted.apply(lambda x: [stemmer.stem(y) for y in x]).str.join(' ')
dataframe['stemmed'] = stemmed_sms

# 3.1 Counting word entries
print('3.1 Counting word entries')
ham_words_count = dataframe[dataframe.type == 'ham'].stemmed.str.split(expand=True).stack().value_counts().reset_index()
ham_words_count.columns = ['word', 'count']

spam_words_count = dataframe[dataframe.type == 'spam'].stemmed.str.split(
    expand=True).stack().value_counts().reset_index()
spam_words_count.columns = ['word', 'count']

# 3.2 Saving words entries to files
print('3.2 Saving words entries to files')
ham_words_count.to_csv('ham_words_count.csv', index=False, header=False)
spam_words_count.to_csv('spam_words_count.csv', index=False, header=False)

# 4.1 Plotting words length distribution and average word length
print('4.1 Plotting words length distribution and average word length')
length_ham = {}
length_spam = {}

for [word, count] in ham_words_count.values:
    length_ham[word] = len(word)

for [word, count] in spam_words_count.values:
    length_spam[word] = len(word)

average_length_ham = round(sum(length_ham.values()) / len(length_ham), 2)
average_length_spam = round(sum(length_spam.values()) / len(length_spam), 2)

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Words length distribution')

ax1.hist(length_ham.values(), color='blue', edgecolor='black', bins=15)
ax1.set_ylabel('Ham words')
ax1.text(20, 1000, f'Average length - {average_length_ham}')

ax2.hist(length_spam.values(), color='blue', edgecolor='black', bins=15)
ax2.set_xlabel('Word length')
ax2.set_ylabel('Spam words')
ax2.text(20, 500, f'Average length - {average_length_spam}')

plt.show()

# 4.2 Plotting messages length distribution and average message length
print('4.2 Plotting messages length distribution and average message length')
length_ham_msg = {}
length_spam_msg = {}

for msg in dataframe[dataframe.type == 'ham'].stemmed:
    length_ham_msg[msg] = len(msg)

for msg in dataframe[dataframe.type == 'spam'].stemmed:
    length_spam_msg[msg] = len(msg)

average_length_ham_msg = round(sum(length_ham_msg.values()) / len(length_ham_msg), 2)
average_length_spam_msg = round(sum(length_spam_msg.values()) / len(length_spam_msg), 2)

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.suptitle('Messages length distribution')

ax1.hist(length_ham_msg.values(), color='blue', edgecolor='black', bins=15)
ax1.set_ylabel('Ham messages')
ax1.text(200, 1000, f'Average length - {average_length_ham_msg}')

ax2.hist(length_spam_msg.values(), color='blue', edgecolor='black', bins=15)
ax2.set_xlabel('Message length')
ax2.set_ylabel('Spam messages')
ax2.text(0, 60, f'Average length - {average_length_spam_msg}')

plt.show()

# 4.3 Plotting top-20 words
print('4.3 Plotting top-20 words')
top_ham_words_count = ham_words_count.sort_values('count', ascending=False)[:20]
top_spam_words_count = spam_words_count.sort_values('count', ascending=False)[:20]

plt.figure(figsize=(11, 4))
plt.title("Top-20 words frequency")
plt.bar(top_ham_words_count['word'], top_ham_words_count['count'])
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()
