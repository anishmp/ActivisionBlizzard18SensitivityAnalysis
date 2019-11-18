# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
import pickle

# Scrapes transcript data from scrapsfromtheloft.com
def url_to_statements(url):
    '''Returns transcript data specifically from Security Exchange Commision Website'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find_all('font')]
    return text

# URLs of transcripts in scope
urls = 'https://www.sec.gov/Archives/edgar/data/718877/000104746918001114/a2234634z10-k.htm'

# Company Name
company = ['Activision Blizzard']

# Actually request transcripts (takes a few minutes to run)
statements = [url_to_statements(urls)]
 
data = {}
for i, c in enumerate(company):
    with open("C:/Users/anish/Documents/CSCE 625/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)

def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

data_combined = {key: [combine_text(value)] for (key, value) in data.items()}

#print(data_combined)

import pandas as pd
pd.set_option('max_colwidth',150)

data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['Company']
data_df = data_df.sort_index()
#print(data_df)

# Apply a first round of text cleaning techniques
import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', '',text)
    text = re.sub('\xa0', '',text)
    text = re.sub('\x97', '',text)
    text = re.sub('\x95', '',text)
    text = re.sub('ý', '',text)
    return text

round1 = lambda x: clean_text_round1(x)

data_clean = pd.DataFrame(data_df.Company.apply(round1))
#print(data_clean.to_dict())
#print(data_clean)


# Let's pickle it for later use
data_clean.to_pickle("corpus.pkl")

# We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.Company)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index



# Let's pickle it for later use
data_dtm.to_pickle("dtm.pkl")

data_dtm = data_dtm.transpose()

top_dict= {}
for c in data_dtm.columns:
    top = data_dtm[c].sort_values(ascending=False).head(30)
    top_dict[c] = list(zip(top.index, top.values))

#print(pd.DataFrame.from_dict(top_dict))


data = pd.read_pickle('corpus.pkl')

from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity


data['polarity'] = data['Company'].apply(pol)
data['subjectivity'] = data['Company'].apply(sub)

import matplotlib.pyplot as plt

#plt.rcParams['figure.figsize'] = [10, 8]

#for index, Company in enumerate(data.index):
#    x = data.polarity.loc[Company]
#    y = data.subjectivity.loc[Company]
#    plt.scatter(x, y, color='blue')
#    plt.text(x+.001, y+.001, "Activision Blizzard", fontsize=10)
#    plt.xlim(-.01, .12) 
    
#plt.title('Sentiment Analysis', fontsize=20)
#plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
#plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

#plt.show()


# Split each routine into 10 parts
import numpy as np
import math

def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)
    
    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

list_pieces = []
for t in data.Company:
    split = split_text(t)
    list_pieces.append(split)  
#print(list_pieces)

polarity_statement = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_statement.append(polarity_piece)
    


# Show the plot for one comedian
plt.plot(polarity_statement[0])
plt.title("Activision Blizzard")
plt.xlabel('Over Time', fontsize=15)
plt.ylabel('<-- Negative -------- Positive -->', fontsize=15)
plt.show()


