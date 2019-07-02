import spacy
from contractions import contractions_dict
import re
from multiprocessing import Pool
import json
class DataPreProcessing():
    """This class will take a list of documents
    as a parameter and return list of processed
    data."""
    nlp = spacy.load('en_core_web_sm')
    noisy_pos_tags = ['-PRON-','-PRON-']
    

    def __init__(self,list_of_doc,lemmatize=True,expand_contraction=True,remove_special_characters=True):
        self()
        self.__list_of_doc=list_of_doc
        self._remove_special_characters=remove_special_characters
        self._lemmatize=lemmatize
        self._expand_contraction=expand_contraction        
    

    def normalized_corpus(self):
        return self.processData()
    
    
    @staticmethod
    def Noise(token):     
        is_noise = False
        if token.is_stop == True:
            is_noise = True
        elif token.pos_ in DataPreProcessing.noisy_pos_tags:
            is_noise = True 
        elif token.string.strip() in DataPreProcessing.noisy_pos_tags:
            is_noise = True
        return is_noise

    def __call__(self):
        add_List=['server']
        remove_List=['not','no']
        for w in add_List:
            DataPreProcessing.nlp.vocab[w].is_stop = True

        for w in remove_List:
            DataPreProcessing.nlp.vocab[w].is_stop = False

    # # Expanding Contractions
    @staticmethod
    def expand_contractions(text,regex,contraction_mapping):
        #print(text)
        match_list=regex.findall(text)
        if len(match_list)>0:
            for word in match_list:
                text=re.sub(word,contraction_mapping.get(word),text)
        return text

    # # Removing Special Characters
    @staticmethod
    def remove_special_characters(text):
        text = re.sub('[^$@a-zA-Z0-9\s]', '', text)
        return text
    
    def processData(self):
        corpus=[]
        contraction_mapping={k.lower(): v.lower() for k, v in contractions_dict.items()}
        ####correcting expansions of some keys
        contraction_mapping["they'd've"]="they would have"
        contraction_mapping["couldn't've"]="could not have"
        contraction_mapping["y’all’d"]="you all would"
        contraction_mapping["we’ll’ve"]="we will have"
        contraction_mapping["i'd've"]="I would have"
        contraction_mapping["wouldn’t’ve"]="would not have"
        contraction_mapping["mustn’t’ve"]="must not have"
        contraction_mapping["won’t’ve"]="will not have"
        contraction_mapping["he'd've"]="he would have"
        contraction_mapping["mightn't've"]="might not have"
        contraction_mapping["y’all’re"]="you all are"
        contraction_mapping["oughtn’t’ve"]="ought not have"
        contraction_mapping["it’ll’ve"]="it will have"
        contraction_mapping["who’ll’ve"]="who will have"
        contraction_mapping["hadn’t’ve"]="had not have"
        contraction_mapping["she’d’ve"]="she would have"
        contraction_mapping["oughtn't've"]="ought not have"
        contraction_mapping["there'd've"]="there would have"
        contraction_mapping["y’all’ve"]="you all have"
        contraction_mapping["mightn’t’ve"]="might not have"
        contraction_mapping["shouldn't've"]="should not have"
        contraction_mapping["how'd'y"]="how did you"
        contraction_mapping["i’ll’ve"]="i will have"
        contraction_mapping["y'all've"]="you all have"
        contraction_mapping["mustn’t’ve"]="must not have"
        contraction_mapping["she’ll’ve"]="she will have"
        contraction_mapping["they’ll’ve"]="they will have"
        contraction_mapping["shouldn’t’ve"]="should not have"
        contraction_mapping["oughtn’t’ve"]="ought not have"
        contraction_mapping["shan't've"]="shall not have"
        contraction_mapping["it’ll’ve"]="it will have"
        contraction_mapping["shan’t’ve"]="shall not have"
        contraction_mapping["who’ll’ve"]="who will have"
        contraction_mapping["hadn’t’ve"]="had not have"
        contraction_mapping["needn’t’ve"]="need not have"
        contraction_mapping["mightn’t’ve"]="might not have"
        contraction_mapping["shouldn't've"]="should not have"
        contraction_mapping["how'd'y"]="how did you"
        contraction_mapping["i’d’ve"]="i would have"
        contraction_mapping["we'll've"]="we will have"
        contraction_mapping["he'll've"]="he will have"
        contraction_mapping["wouldn't've"]="would not have"
        contraction_mapping["we’d’ve"]="we would have"
        contraction_mapping["can't've"]="cannot have"
        contraction_mapping["couldn’t’ve"]="could not have"
        contraction_mapping["i'll've"]="i will have"
        contraction_mapping["what’ll’ve"]="what will have"
        contraction_mapping["y’all’d’ve"]="you all would have"
        contraction_mapping["y'all're"]="you all are"
        contraction_mapping["there’d’ve"]="there would have"
        contraction_mapping["he’d’ve"]="he would have"
        contraction_mapping["you'd've"]="you would have"
        contraction_mapping["there’d’ve"]="there would have"
        contraction_mapping["he’ll’ve"]="he will have"
        contraction_mapping["will've"]="will have"
        contraction_mapping["cannot’ve"]="cannot have"
        contraction_mapping["you will've"]="you will have"
        contraction_mapping["he will’ve"]="he will have"
        contraction_mapping["will not've"]="will not have"
        contraction_mapping["he would’ve"]="he would have"
        contraction_mapping["all’ve"]="all have"
        """
        print("***********************************************************************************************************************************")
        with open("contraction.json","w") as f1:
            json.dump(contraction_mapping,f1,ensure_ascii=True)

        with open("contraction.json","r") as f1:
            abc=json.loads(f1.read())
            print(abc.items())
        print("***********************************************************************************************************************************")
        """
        regex = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
        for doc in self.__list_of_doc:
            if self._expand_contraction:
                doc=self.expand_contractions(doc.strip().lower(),regex,contraction_mapping)

            if self._remove_special_characters:
                doc=self.remove_special_characters(doc)
                
            document=DataPreProcessing.nlp(doc.lower())
            if not self._lemmatize:
                cleaned_list = " ".join([token.string.strip() for token in document if not self.Noise(token) and len(token.string.strip())>0])
                corpus.append(cleaned_list)
            else:
                cleaned_list = " ".join([token.lemma_.strip() for token in document if not self.Noise(token) and len(token.string.strip())>0])
                corpus.append(cleaned_list)

        return corpus

def main(data):
    from os import getpid
    d=DataPreProcessing(data)
    corpus=d.normalized_corpus()
    return corpus
    
    

if __name__=="__main__":
    data=["how to ma'am resolve error $for# it'd not Server he will’ve Load",'@ How all’ve to resolving error for No file found','job abended or failed  for script','how to executing a script','efgh ',\
      'Hi']
    print(main(data))
        
        
    
    
    
            
