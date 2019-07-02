import spacy 
nlp = spacy.load('en_core_web_sm')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from DataProcessing import DataPreProcessing,main
id=[1,2,3,4]
data=['how to resolve error for Server Load','How to resolve error for No file found','job abended or failed  for script','how to execute a script','efgh ',\
      'Hi']
target=['run top to idntify the process.kill it ','specify the correct directory','check the logs','execute a sh/source','abcd','Hey Let me know how could I help you']
test=['abcd']

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




def question(corpus):
    print("a")
    tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2),sublinear_tf=True)
    tfidf_matrix_train = tv.fit_transform(corpus)
    #tfidf_matrix_train=tfidf_matrix_train.toarray()
    #vocab=tv.get_feature_names()
    #df=pd.DataFrame(tfidf_matrix_train,columns=vocab)
    #print(df)
    return tv, tfidf_matrix_train

def reply(test,tv,tfidf_matrix_train):
    #print(tv)
    print("b")
    test=(test,)
    tfidf_matrix_test = tv.transform(test)
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    #print(cosine)


    minimum_score=0.3
    #cosine = np.delete(cosine, 0)  #not required
    #print(cosine)
    maxa = cosine.max()
    #print(maxa)
    response_index=-99999
    if (maxa >= minimum_score):
        #print("hello")
        #new_max = maxa - 0.01
        alist = np.where(cosine > minimum_score)[1]
        #alist = np.where(cosine > new_max)
        # print ("number of responses with 0.01 from max = " + str(list[0].size))
        #response_index = random.choice(alist[0])
        #print(alist)
        #print(response_index)
        for index in alist:
            return target[index]
    

 
 

if __name__=="__main__":
    #d=DataPreProcessing(data)
    #corpus=d.normalized_corpus()
    corpus=main(data)
    tv, tfidf_matrix_train=question(corpus)
    while True:
        ques=input("rudra:" )
        ques=(ques,)
        ques=" ".join(main(ques))
        print(ques)
        print(reply(ques,tv,tfidf_matrix_train))
