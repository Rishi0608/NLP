import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd
import matplotlib.pyplot as plt
en_stop = stopwords.words('english')
en_stop.remove('not')
data = pd.read_csv("/Users/rishijangid/Downloads/tripadvisor_hotel_reviews.csv")
data['Review_Preprocessed'] = data['Review'].str.lower()
data['Review_noStopword'] = data['Review_Preprocessed'].apply(lambda x:" ".join([i for i in x.split() if i not in (en_stop)]))
data['No_Punc'] = data.apply(lambda x:re.sub(r"[*]", "star", x["Review_noStopword"]), axis=1)
data['No_Punc'] = data.apply(lambda x: re.sub(r"([^\w\s])", "",x['No_Punc']), axis=1)
data['Tokenized'] = data.apply(lambda x:word_tokenize(x["No_Punc"]), axis=1)
ps = PorterStemmer()
data["Stemmed"] = data['Tokenized'].apply(lambda x: [ps.stem(i)for i in x])
lemmatizer = WordNetLemmatizer()
data['lemmatized'] = data['Tokenized'].apply(lambda x: [lemmatizer.lemmatize(i) for i in x])
token_clean = sum(data['lemmatized'], [])
ngram = (pd.Series(nltk.ngrams(token_clean,4)).value_counts())
ngram[:10].sort_values().plot.bar(color="red", width = .9, figsize = (12,8))
plt.title("10 most relevant words")
plt.xlabel("most relevant words")
