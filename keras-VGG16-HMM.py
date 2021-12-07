import time
a=time.time()
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
b=time.time()
from os import listdir
from os.path import isfile, join
import PIL
import pandas as pd
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import inaugural
from nltk.corpus import abc
from nltk.corpus import brown
import nltk.corpus
#nltk.download('punkt')
#nltk.download('movie_reviews')
#nltk.download('inaugural')
#nltk.download('abc')
#nltk.download('brown')

import numpy as np
#import glob
import random
from random import random
#from tensorflow import coco_captions

print("Done w tensorflow")


def getFiles(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #print(onlyfiles[0:10])
    print(len(onlyfiles))
    return onlyfiles

def identifications(pics):    
    model = VGG16()
    totLabels=""
    numWritten=0
    a=time.time()
    f=open('vallabels.txt', 'a+')
    for file in pics:
        curFile="coco/images/val2014/"+file
        image = PIL.Image.open(curFile).resize((224,224))
        width, height = image.size
        #image = image.resize((224,224))
        curImage = load_img(curFile, target_size=(height,width))
        
        image = img_to_array(curImage)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        yhat = model.predict(image)
        label = decode_predictions(yhat)
        label = label[0][0][1]
        
        numWritten+=1
        if numWritten%100==0:
            print("done w ",numWritten)
            print("took ",time.time()-a)
            a=time.time()
        #print(totLabels)
        
        f.write(label)
        f.write("\n")
    f.close()
#identifications(getFiles("coco/images/val2014"))


    
def most_likely_word_after(aWord,next_word_matrix,word_idx_dict,distinct_words):
    most_likely = next_word_matrix[word_idx_dict[aWord]].argmax()
    return distinct_words[most_likely]

def naive_chain(seed, next_word_matrix,word_idx_dict,distinct_words,length=5):
    current_word = seed
    sentence = seed

    for _ in range(length):
        sentence+=' '
        next_word = most_likely_word_after(current_word,next_word_matrix,word_idx_dict,distinct_words)
        sentence+=next_word
        current_word = next_word
    return sentence

def sample_next_word_after(aWord, next_word_matrix, word_idx_dict, distinct_words, alpha = 0):
    if aWord in word_idx_dict:
        next_word_vector = next_word_matrix[word_idx_dict[aWord]] + alpha
    else:
        next_word_vector = next_word_matrix[word_idx_dict["object"]] + alpha
    likelihoods = next_word_vector/next_word_vector.sum()
    return weighted_choice(distinct_words, likelihoods)

def weighted_choice(objects, weights):
    """ returns randomly an element from the sequence of 'objects', 
        the likelihood of the objects is weighted according 
        to the sequence of 'weights', i.e. percentages."""
    weights = np.array(weights, dtype=np.float64)
    sum_of_weights = weights.sum()
    # standardization:
    np.multiply(weights, 1 / sum_of_weights, weights)
    weights = weights.cumsum()
    x = random()
    for i in range(len(weights)):
        if x < weights[i]:
            return objects[i]
        
def stochastic_chain(word, next_word_matrix, word_idx_dict, distinct_words, length=10):
    current_word = word
    sentence = word

    for _ in range(length):
        sentence+=' '
        next_word = sample_next_word_after(current_word, next_word_matrix, word_idx_dict, distinct_words)
        sentence+=next_word
        current_word = next_word
    return sentence
        
def hmm_model(files,classifications,outFile):
    captions=""
    #read in classifications, sort
    f=open(classifications,"r")
    classifications=f.read().split("\n")
    for i in range(len(classifications)):
        if "_" in classifications[i]:
            cur_word=classifications[i]
            classifications[i]=cur_word.replace("_"," ")
    #get sentences
    minLength=5
    sentences=list(movie_reviews.sents())
    sentences.extend(list(inaugural.sents()))
    sentences.extend(list(brown.sents()))
    sentences.extend(list(abc.sents()))
    print(sentences[-2:])
    print(type(sentences))
    print(len(sentences))
    print(sentences[0:5])
    #get corpus for sentence
    corpus = ""
    for sentence in sentences:
        sentence=sentence[2:]
        for word in sentence:
            corpus+=word
            corpus+=" "
            #print(corpus)
    corpus = corpus.replace('\n',' ')
    corpus = corpus.replace('\t',' ')
    corpus = corpus.replace('“', ' " ')
    corpus = corpus.replace('”', ' " ')
    for spaced in ['.','-',',','!','?','(','—',')']:
        corpus = corpus.replace(spaced, ' {0} '.format(spaced))
    corpus_words = corpus.split(' ')
    corpus_words= [word for word in corpus_words if word != '']
    #print(corpus_words[0:20])
    distinct_words = list(set(corpus_words))
    word_idx_dict = {word: i for i, word in enumerate(distinct_words)}
    #for i in word_idx_dict:
    #    print(i,word_idx_dict[i])
    #print(word_idx_dict)
    distinct_words_count = len(list(set(corpus_words))) #how many unique words
    #distinct_words_count
    next_word_matrix = np.zeros((distinct_words_count,distinct_words_count))
    for i, word in enumerate(corpus_words[:-1]):
        first_word_idx = word_idx_dict[word]
        next_word_idx = word_idx_dict[corpus_words[i+1]]
        next_word_matrix[first_word_idx][next_word_idx] +=1
    print(naive_chain("the",next_word_matrix,word_idx_dict,distinct_words))
    print(sample_next_word_after('ostrich',next_word_matrix,word_idx_dict,distinct_words))
    result=[]
    start=time.time()
    a=0
    print("diff lens?",len(classifications),len(files))
    for i in range(len(classifications)):
        word=classifications[i]
        curDic={}
        curDic["image_id"]=files[i].split(".")[0].split("_")[-1]
        #print(curDic["image_id"])
        
        if " " in word: 
            curDic["caption"]=stochastic_chain(word.split(" ")[-1],next_word_matrix, word_idx_dict, distinct_words)
        else:
            curDic["caption"]=stochastic_chain(word, next_word_matrix, word_idx_dict, distinct_words)
        result.append(curDic)
        a+=1
        
        if a%100==0:
            print("done",time.time()-start)
            start=time.time()
    f=open(outFile,"w+")
    f.write(result)
    f.close() 
    return result
hmm_model(getFiles("coco/images/train2014"),"train.txt","trainCaptions.txt")
