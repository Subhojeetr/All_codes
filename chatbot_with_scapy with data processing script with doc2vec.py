from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from DataProcessing import DataPreProcessing,main
import json
import pandas as pd 

data=['how to resolve error for Server Load','How to resolve error for No file found','job abended or failed  for script','how to execute a script',
      'He said to told you']
target=['run top to idntify the process.kill it ','specify the correct directory','check the logs','execute a sh/source','Hey Let me know how could I help you']


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
data=q[:4000]
target=a[:4000]

del a
del q


def modelTraining(norm_corpus,model_nm):

    tagged_data = [TaggedDocument(words=doc.split(' '), tags=[str(i)]) for i, doc in enumerate(norm_corpus)]
    #print(tagged_data)

    max_epochs = 100
    vec_size = 5
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
    model.build_vocab(tagged_data)

    for epoch in range(max_epochs):
        #print('iteration {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,epochs=1)          #model.iter)
        # decrease the learning rate
        model.alpha -= 0.0001
        # fix the learning rate, no decay
        #model.min_alpha = model.alpha

    model.save(model_nm)
    print("Model Saved")

def prediction(model,question):

    model= Doc2Vec.load(model)
    #to find the vector of a document which is not in training data
    test_data = question
    print(question)
    model.random.seed(0)
    v1 = model.infer_vector(test_data)
    print("V1_infer", v1)
    #model.similarity(v1,v1)
    sims = model.docvecs.most_similar([v1])
    for i in range(len(sims)):
        #tagged_data[int(sims[i][0])].words
        print("################################################################################")
        cosine=sims[i][1]
        print("cosine="+str(cosine))
        n=int(sims[i][0])
        print(data[n])
        print(n)
        print(target[n])
        print("################################################################################")
        #print(" ".join(tagged_data[int(sims[i][0])].words))

if __name__=="__main__":
    norm_corpus=main(data)
    modelTraining(norm_corpus,"d2v.model")
    while True:
        ques=input("rudra:" )
        question=main((ques,))
        prediction("d2v.model",question)
