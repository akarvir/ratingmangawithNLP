

'''dataframe = pd.read_csv('fifa_players.csv')
dataframe = dataframe.iloc[:, :10]
print(dataframe.head())
dataframe.to_csv('output.csv',index=False)'''

import ssl
import praw
import markdown
from bs4 import BeautifulSoup
import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')




# That's it? This is gonna be a very small project.

client_id = "HAFhbjN8k7Vp3G7r4VfDOw"
secret_key = "JfCDyZhP0-FwZye0R0K-NxmxMns8Mw"
user_agent = "User-Agent: sentimentanalysisofmanga/1.0 by u/Common_Purchase3165"
n = 271 # final jjk chapter number. Any submission has this in its title, its sure to be referencing the chapter. But man is it gonna be confusing.


#Instantiate a Reddit Instance

# For actual chapter discussion threads, Jujutsushi has discussions for every chapter dating back with keyword "Discussion"
# Leaks LEAKS leaks

reddit = praw.Reddit(client_id = client_id, client_secret = secret_key, user_agent = user_agent)

potential_subreddit = "JuJutsuKaisen"
anotherpotentialsubreddit = "Jujutsufolk"
andanothersubreddit = "Jujutsushi"


subreddit = reddit.subreddit(anotherpotentialsubreddit)


# storing all potential threads

chapter_threads = []
for thread in subreddit.top(limit = 200):
    if "LEAKS" in thread.title or "leaks" in thread.title or "Leaks" in thread.title:

        chapter_threads.append(thread)


all_comments = []

for thread in chapter_threads: 

      thread.comments.replace_more(limit=0)
      for comment in list(thread.comments):
        
        all_comments.append(comment.body)



tempsentscores = [0]*len(all_comments)

df = pd.DataFrame({'comment':all_comments,'sentiment_score':tempsentscores})


def preprocess_text(text):

    tokens = word_tokenize(text.lower())

    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    processed_text = ''.join(lemmatized_tokens)

    return processed_text

df['comment'] = df['comment'].apply(preprocess_text)

analyser = SentimentIntensityAnalyzer()


def get_sentiment(text): 

    scores = analyser.polarity_scores(text) # returns dictionary of positive score, negative score, and neutral score. We'll check for positive. 
    sentiment = 1 if scores['pos'] > 0 else -1
    return sentiment


df['sentiment_score'] = df['comment'].apply(get_sentiment)

print(df.head)







# building the model, or rather assigning a value to each one of em
