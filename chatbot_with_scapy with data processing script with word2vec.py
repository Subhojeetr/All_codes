import spacy 
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy import spatial
import numpy as np
import json
from DataProcessing import DataPreProcessing,main
id=[1,2,3,4]
data=['how to resolve error for Server Load','How to resolve error for No file found','job abended or failed  for script','how to execute a script','efgh ',\
      'Hi']
target=['run top to idntify the process.kill it ','specify the correct directory','check the logs','execute a sh/source','abcd','Hey Let me know how could I help you']


q=[]
a=[]
print(len(a))
with open('train.json', "r") as sentences_file:
    reader = json.load(sentences_file)
    for item in reader['data']:
        if type(item)==dict:
            for qas in item['paragraphs']:
                for question in qas['qas']:
                    try:
                        a.append(question['answers'][0]['text'])
                        q.append(question['question'])
                        
                    except:
                        pass
                    break
q.extend(data)
a.extend(target)
print(len(a))
data=q
target=a
del a
del q



from gensim.models import word2vec
import nltk
import pandas as pd

from scipy import spatial

wpt = nltk.WordPunctTokenizer()
def vectorize(norm_corpus):
    tokenized_corpus = [wpt.tokenize(document) for document in norm_corpus]
    print(tokenized_corpus)

    # Set values for various parameters
    feature_size = 10 # Word vector dimensionality
    window_context = 10 # Context window size
    min_word_count = 1 # Minimum word count
    sample = 1e-3 # Downsample setting for frequent words
    w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size,window=window_context, min_count = min_word_count,sample=sample)
    w2v_model.train(tokenized_corpus, total_examples=len(tokenized_corpus), epochs=10)
    #print(w2v_model.wv['script'])

    def average_word_vectors(words, model, vocabulary, num_features):
        feature_vector = np.zeros((num_features,),dtype="float64")
        nwords = 0.
        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, model[word])
        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    def averaged_word_vectorizer(corpus, model, num_features):
        vocabulary = set(model.wv.index2word)
        features = [average_word_vectors(tokenized_sentence, model, vocabulary,num_features) for tokenized_sentence in corpus]
        return np.array(features)

    w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model,num_features=feature_size)
    print(pd.DataFrame(w2v_feature_array))
    return w2v_feature_array

def calSpatialCosine(test_vector,train_vector):
    for train in train_vector:
        cosine=1-spatial.distance.cosine(test_vector[0],train)
        print(cosine)

def calCosineSimilarity(test_vector,train_vector):
    minimum_score=0.3
    cosine = cosine_similarity(test_vector, train_vector)[0][1:]
    print(cosine)
    
    minimum_score=0.3
    maxa = cosine.max()
    print(maxa)
    if (maxa >= minimum_score):
        #print("hello")
        #new_max = maxa - 0.01
        alist = np.where(cosine > minimum_score)
        print(alist)

        for index in alist[0]:
                print("**************")
                print(index)
                print(target[index])
                print(data[index])
                print("**************")

        blist=np.where(cosine==maxa)[0]
        print("******MAX b********")
        print(blist)
        print("Actual reply")
        print(target[blist[0]])
        print("******MAX b********")
    

if __name__=="__main__":
    #d=DataPreProcessing(data)
    #corpus=d.normalized_corpus()
    norm_train_corpus=main(data)
    train_vector=vectorize(norm_train_corpus)
    print("##############################################################################################################")
    question=['how to resolve error for Server Load']
    while True:
        question=input("rudra:" )
        ques=(question,)
        norm_test_corpus=main(ques)
        test_vector=vectorize(norm_test_corpus)
        print("##############################################################################################################")
        calCosineSimilarity(test_vector,train_vector)
        print("##############################################################################################################")
        #calSpatialCosine(test_vector,train_vector)
        print("##############################################################################################################")
        #print(corpus)    
    
