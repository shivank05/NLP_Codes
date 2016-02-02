
# coding: utf-8

# In[845]:

get_ipython().magic(u'matplotlib qt')
from collections import Counter
import nltk
import nltk.metrics
from collections import defaultdict
import hashlib
import io
import math
import matplotlib.pyplot as plt
import os
import re
import tarfile
from urllib import urlretrieve
from nltk.tokenize import word_tokenize,sent_tokenize



# In[846]:

""" get_files will access the files from the mentioned path and will append the path address to the list which will be used 
to train and test data """
def get_files(path):
    file_list=[]
    for files in os.listdir(path):
        if files.endswith(".dp"):
            files = os.path.join(path,files)
            file_list.append(files)
    return sorted(file_list)
lst = get_files("C:\Users\Shivank\Desktop\NLP\Assignments\Assignment01\dependency_treebank")


# In[847]:

"""The tokenize_tag function will create the tag the words in each sentence of every file and will create the tuple 
in form of (word,tag) . The function will create the list of the (word,tag) which will be used to measure overall accuracy 
, accuracy per sentence , precision and Recall """
def tokenize_tag(lst):
    k=[]
    final=[]
    for files in lst:
        with open(files) as f:
            for line in f:
                if line !="\n":
                    st=(line.split()[0]+'/'+line.split()[1])
                    tagged_token = nltk.tag.str2tuple(st)
                    k.append(tagged_token)
                    if tagged_token[0] == '.':
                        final.append(k)
                        k=[]
                    
    return final
p=[]
p=tokenize_tag(lst)


# In[848]:

size = int(len(p) * 0.9)


# In[849]:

train_sents = p[:size]
test_sents = p[size:]
"""The overall_accuracy function will display the accuracy of the UnigramTagger , Bigram_tagger , Uni-Bi-Tri backoff tagger 
and HMM Tagger"""

def overall_accuracy(train_sents,test_sents):
    unigram_tagger = nltk.UnigramTagger(train_sents)
    a1=unigram_tagger.evaluate(test_sents)
    Bigram_tagger = nltk.BigramTagger(train_sents)
    b1=Bigram_tagger.evaluate(test_sents)
    t0 = nltk.DefaultTagger('NN')
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    c1=t2.evaluate(test_sents)
    nltk_HMM = nltk.tag.hmm.HiddenMarkovModelTrainer() 
    train_HMM = nltk_HMM.train_supervised(train_sents, estimator = nltk.LaplaceProbDist)
    d1=train_HMM.evaluate(test_sents)
    return a1,b1,c1,d1 
lst_overall_accuracy = overall_accuracy(train_sents,test_sents)
print lst_overall_accuracy


# In[850]:

"""The below code will be find the Average accuracy per sentence using the different Tagger Functions
( Unigram Tagger , Bigram Tagger , Uni-Bi-Tri Backoff tagger) """
accuracy_sentence=[]
accuracy=[]
for l in test_sents:
    accuracy.append(unigram_tagger.evaluate([l]))
    s=sum(accuracy)/len(test_sents)
print "Average accuracy per sentence using Unigram "
print s
accuracy_sentence.append(s)
accuracy=[]
for l in test_sents:
    accuracy.append(Bigram_tagger.evaluate([l]))
    s=sum(accuracy)/len(test_sents)
print "Average accuracy per sentence using Bigram "
print s
accuracy_sentence.append(s)
accuracy=[]
for l in test_sents:
    accuracy.append(t2.evaluate([l]))
    s=sum(accuracy)/len(test_sents)
print "Average accuracy per sentence using UNIBITRI Backoff "
print s
accuracy_sentence.append(s)
accuracy=[]
for l in test_sents:
    accuracy.append(train_HMM.evaluate([l]))
    s=sum(accuracy)/len(test_sents)
print "Average accuracy per sentence using HMM "
print s
accuracy_sentence.append(s)


# In[851]:

a=[]
true_set =[]
for i in test_sents:
    for k in i:
        a.append(k[0])
        true_set.append(k)
        
list_of_tags=p
s=[]
for i in list_of_tags:
    for temp in i:
        s.append(temp[1])
final_list = list(set(s))


# In[852]:

def precision_recall(final_list,true_set,Ref_Set):
    
    sim1=[]
    sim2=[]
    sim3=[]
    g=len(true_set)
    
    for t in final_list:
        truepos=0.0
        falseneg=0.0
        falsepos=0.0
        trueneg= 0.0
        for i in range(len(true_set)):
            if true_set[i][1] == t and Ref_Set[i][1] == t:
                truepos=truepos+1
            elif true_set[i][1] != t and Ref_Set[i][1] == t:
                falseneg=falseneg+1
            elif true_set[i][1]== t and Ref_Set[i][1] != t:
                falsepos= falsepos+1
            elif true_set[i][1]!= t and Ref_Set[i][1] != t:
                trueneg = trueneg+1
        if (truepos+falsepos) >0 :
            precision = truepos/(truepos+falsepos)
            sim1.append(precision)
        if(truepos+falseneg) > 0:       
            recall = truepos/(truepos+falseneg)
            sim2.append(recall)
        acc_g1 = (truepos+trueneg)/g
        tu_g1 = (acc_g1,t)
        sim3.append(tu_g1)
        

    average_precision = sum(sim1)/len(sim1)
    average_recall = sum(sim2)/len(sim2)
    return average_precision,average_recall,sim3

print lst_overall_accuracy#overall accuracy of the taggers
print accuracy_sentence#average accuracy per sentence for the taggers


# In[853]:

"""graph to Compare the overall accuracies for the Unigram , Bigram , UniBiTri backoff and HMM tagger"""
import pylab

names = ['Unigram','Bigram','UniBiTri_Backoff','HMM']

pylab.figure(1)
x = range(4)
pylab.xticks(x, names)
pylab.plot(x,lst_overall_accuracy,"bo-")

pylab.show()


# In[854]:

"""Graph to compare the average accuracies per sentence using Unigram , Bigram , UniBiTri and HMM"""

names = ['Unigram','Bigram','UniBiTri_Backoff','HMM']

pylab.figure(1)
x = range(4)
pylab.xticks(x, names)
pylab.plot(x,accuracy_sentence,"ro-")

pylab.show()


# In[855]:

Unigram_t = precision_recall(final_list,true_set,Ref_Set1)
print "Precision for UniGram Tagger is " 
print Unigram_t[0]
print "Recall for UniGram Tagger is " 
print Unigram_t[1]
lst1=[]
lst2=[]
l1=[]
print " The Tags with highest accuracy for Unigram Tagger are :"
for i in Unigram_t[2]:
    lst1.append(i[0])
    lst2.append(i[1])
    if i[0]==1.0:
        l1.append(i[1])
print l1

pylab.figure(1)
x = range(len(lst1))
pylab.xticks(x, lst2)
pylab.plot(x,lst1,"ro-")
pylab.show()



# In[856]:

Bigram_t = precision_recall(final_list,true_set,Ref_Set2)
print "Precision for BiGram Tagger is " 
print Bigram_t[0]
print "Recall for BiGram Tagger is " 
print Bigram_t[1]
lst1=[]
lst2=[]
l1=[]
print " The Tags with highest accuracy for Bigram Tagger are :"
for i in Bigram_t[2]:
    lst1.append(i[0])
    lst2.append(i[1])
    if i[0]==1.0:
        l1.append(i[1])
print l1

pylab.figure(1)
x = range(len(lst1))
pylab.xticks(x, lst2)
pylab.plot(x,lst1,"ro-")
pylab.show()




    


# In[857]:

UNIBITRI_t = precision_recall(final_list,true_set,Ref_Set3)
print "Precision for UniBiTriBackOff Tagger is " 
print UNIBITRI_t[0]
print "Recall for UniBiTriBackoff Tagger is " 
print UNIBITRI_t[1]
lst1=[]
lst2=[]
l1=[]
print " The Tags with highest accuracy for UniBiTriBackoff Tagger are :"
for i in UNIBITRI_t[2]:
    lst1.append(i[0])
    lst2.append(i[1])
    if i[0]==1.0:
        l1.append(i[1])
print l1

pylab.figure(1)
x = range(len(lst1))
pylab.xticks(x, lst2)
pylab.plot(x,lst1,"ro-")
pylab.show()


# In[858]:

Hmm_t = precision_recall(final_list,true_set,Ref_Set4)
print "Precision for HMM Tagger is " 
print Hmm_t[0]
print "Recall for HMM Tagger is " 
print Hmm_t[1]
lst1=[]
lst2=[]
l1=[]
print " The Tags with highest accuracy for HMM Tagger are :"
for i in Hmm_t[2]:
    lst1.append(i[0])
    lst2.append(i[1])
    if i[0]==1.0:
        l1.append(i[1])
print l1

pylab.figure(1)
x = range(len(lst1))
pylab.xticks(x, lst2)
pylab.plot(x,lst1,"ro-")
pylab.show()

