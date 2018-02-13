
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats

import os
import subprocess
import sys
import shutil

import codecs


# In[ ]:


import matlab.engine
eng = matlab.engine.start_matlab()    


# # [Lee95]: Combining Multiple Evidence from Different Properties of Weighting Schemes

# In[3]:


def normalize_score (path, filename_in, dir_in, dir_out):
    path_in = path + "\\" + dir_in + "\\" + filename_in
    df = pd.read_csv(path_in, delimiter = " ", header = None)
    df.columns = ["topicID", "q0", "docID" , "rank", "score" , "model"]

    summary = stats.describe(df[:]['score'])
    minimum = summary[1][0]
    maximum = summary[1][1]
    norm = np.asarray(df[:]['score'])
    norm = (norm - minimum) / float((maximum - minimum))#qui viene fatta una copia della colonna
    df[:]['score'] = norm

    path_out = path + "\\" + dir_out + "\\n_" + filename_in
    df.to_csv(path_out, index = False, header = False, sep = " ")
    
def normalize_score_all (filename_list, path, dir_in, dir_out):
    for filename in filename_list:
        normalize_score(path, filename, dir_in, dir_out)


# # [FoxShaw93]: Combination of Multiple Searches

# In[4]:


def comb_sum (filename_list, path, dir_in, dir_out,nome_file="comb_sum.txt"):
    comb_sum = {}   
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                comb_sum.setdefault((topicID, documentID), 0)
                comb_sum[(topicID, documentID)] += score

    comb_sum = sorted(comb_sum.items(), key = lambda (k, v) : (v, k), reverse = True)
    comb_sum = np.asarray([list(k) + [v] for k, v in comb_sum])
    
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = comb_sum, columns = ['topicID', 'docID', 'score'])
    df = df.sort_values(['topicID', 'score'], ascending=[True, False])
    df['Rank'] = df.groupby('topicID')['score'].rank(ascending = False).astype('int64') - 1
    df['Q0'] = "Q0"
    df['model'] = "CombSum"
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'Rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ")


# In[5]:


def comb_max (filename_list, path, dir_in, dir_out, nome_file="comb_max.txt"):
    comb_max = {}   
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                comb_max.setdefault((topicID, documentID), 0)
                if(comb_max[(topicID, documentID)] < score):
                      comb_max[(topicID, documentID)] = score
    
    comb_max = sorted(comb_max.items(), key = lambda (k, v) : (v, k), reverse = True)
    comb_max = np.asarray([list(k) + [v] for k, v in comb_max])
    
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = comb_max, columns = ['topicID', 'docID', 'score'])
    df = df.sort_values(['topicID', 'score'], ascending=[True, False])
    df['Rank'] = df.groupby('topicID')['score'].rank(ascending = False).astype('int64') - 1
    df['Q0'] = "Q0"
    df['model'] = "CombMax"
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'Rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ")


# In[6]:


def comb_min (filename_list, path, dir_in, dir_out, nome_file="comb_min.txt"):
    comb_min = {}   
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                comb_min.setdefault((topicID, documentID), 2) # Ipothesis: normalized values
                if(comb_min[(topicID, documentID)] > score):
                      comb_min[(topicID, documentID)] = score  
    
    comb_min = sorted(comb_min.items(), key = lambda (k, v) : (v, k), reverse = True)
    comb_min = np.asarray([list(k) + [v] for k, v in comb_min])
    
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = comb_min, columns = ['topicID', 'docID', 'score'])
    df = df.sort_values(['topicID', 'score'], ascending=[True, False])
    df['Rank'] = df.groupby('topicID')['score'].rank(ascending = False).astype('int64') - 1
    df['Q0'] = "Q0"
    df['model'] = "CombMin"
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'Rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ")


# In[7]:


def comb_median (filename_list, path, dir_in, dir_out, nome_file="comb_median.txt"):
    comb_median = {}   
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                comb_median.setdefault((topicID, documentID), [])
                comb_median[(topicID, documentID)].append(score)
    for k in comb_median:
        median = np.median(np.asarray(comb_median[k]))
        comb_median[k] = median

    comb_median = sorted(comb_median.items(), key = lambda (k, v) : (v, k), reverse = True)
    comb_median = np.asarray([list(k) + [v] for k, v in comb_median])
    
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = comb_median, columns = ['topicID', 'docID', 'score'])
    df = df.sort_values(['topicID', 'score'], ascending=[True, False])
    df['Rank'] = df.groupby('topicID')['score'].rank(ascending = False).astype('int64') - 1
    df['Q0'] = "Q0"
    df['model'] = "CombMedian"
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'Rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ") 


# In[8]:


def comb_mnz (filename_list, path, dir_in, dir_out, nome_file="comb_mnz.txt"):
   comb_mnz = {}   
   for filename_in in filename_list:
       path_in = path + "\\" + dir_in + "\\" + filename_in
       in_file = pd.read_csv(path_in, delimiter = " ", header = None)
       in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
       for i in range(0, in_file.shape[0], 1):
               topicID = in_file['topicID'][i]
               documentID = in_file['docID'][i]
               score = in_file['score'][i]
               comb_mnz.setdefault((topicID, documentID), [0,0])
               comb_mnz[(topicID, documentID)][0] += score
               comb_mnz[(topicID, documentID)][1] += 1   
   for k in comb_mnz:
       comb_mnz[k] = comb_mnz[k][0] * comb_mnz[k][1]
       
   comb_mnz = sorted(comb_mnz.items(), key = lambda (k ,v) : (v, k), reverse = True)
   comb_mnz = np.asarray([list(k[0]) + [k[1]] for k in comb_mnz])
   
   path_out = path + "\\" + dir_out + "\\" + nome_file
   df = pd.DataFrame(data = comb_mnz, columns = ['topicID', 'docID', 'score'])
   df = df.sort_values(['topicID', 'score'], ascending=[True, False])
   df['Rank'] = df.groupby('topicID')['score'].rank(ascending = False).astype('int64') - 1
   df['Q0'] = "Q0"
   df['model'] = "CombMnz"
   df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'Rank', 'score', 'model'])
   df.to_csv(path_out, index = False, header = False, sep = " ")


# In[9]:


def comb_anz (filename_list, path, dir_in, dir_out, nome_file="comb_anz.txt"):
    comb_anz = {}   
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                comb_anz.setdefault((topicID, documentID), [0,0])
                comb_anz[(topicID, documentID)][0] += score
                comb_anz[(topicID, documentID)][1] += 1
    for k in comb_anz:
        comb_anz[k] = comb_anz[k][0] / (float)(comb_anz[k][1])

    comb_anz = sorted(comb_anz.items(), key = lambda (k, v) : (v, k), reverse = True)
    comb_anz = np.asarray([ list(k[0]) + [k[1]] for k in comb_anz])
    
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = comb_anz, columns = ['topicID', 'docID', 'score'])
    df = df.sort_values(['topicID', 'score'], ascending=[True, False])
    df['Rank'] = df.groupby('topicID')['score'].rank(ascending = False).astype('int64') - 1
    df['Q0'] = "Q0"
    df['model'] = "CombAnz"
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'Rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ")    


# # [CF02]: Condorcet Fusion for Improved Retrieval

# In[10]:


def algorithm1(k1, k2):
    count = 0
    for filename in filename_list_global:   
        flag1 = filename in condorcet[k1]
        flag2 = filename in condorcet[k2]
        if (flag1 == True and flag2 == False):
            count = count + 1
        if (flag1 == False and flag2 == True):
            count = count - 1
        if (flag1 == True and flag2 == True):
            if condorcet[k1][filename] > condorcet[k2][filename]:
                count += 1
            else:
                count -= 1
    if(count > 0):
        return 1
    return -1


# In[11]:


def condorcet_alg (filename_list, path, dir_in, dir_out, nome_file="condorcet.txt"):
    global condorcet
    global filename_list_global
    condorcet = {}  
    L = set({})
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                condorcet.setdefault((topicID, documentID), {})
                condorcet[(topicID, documentID)][filename_in] = score
                L.add((topicID,documentID))        
    
    filename_list_global = filename_list
    L = list(L) 
    LL = {}
    for i in L:
        LL.setdefault(i[0], [])
        LL[i[0]].append((i[0], i[1]))
        
    for k in LL:
        LL[k] = sorted(LL[k], cmp = algorithm1, reverse = True)
        LL[k] = [ np.asarray(list(LL[k][i]) + [i]) for i in range(len(LL[k]))]
        
        LL[k] = np.asarray(LL[k])
       
    Matrix=[]
    for k in LL:
        for i in LL[k]:
            Matrix.append(i)
    Matrix=np.asarray(Matrix) 
        
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = Matrix, columns = ['topicID', 'docID', 'rank'])
    df['rank'] = df['rank'].astype('int64')
    df['score'] = df['rank'].max() - df['rank']  #1. / (df['rank'].astype('int64') + 1)
    df = df.sort_values(['topicID', 'rank'], ascending=[True, True])
    df['Q0'] = "Q0"
    df['model'] = "Condorcet"
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ") 


# In[12]:


def algorithm1_weighted(k1, k2):
    count = 0
    for filename in filename_list_global:   
        flag1 = filename in condorcet[k1]
        flag2 = filename in condorcet[k2]
        if (flag1 == True and flag2 == False):
            count = count + w[filename]
        if (flag1 == False and flag2 == True):
            count = count -  w[filename]
        if (flag1 == True and flag2 == True):
            if condorcet[k1][filename] > condorcet[k2][filename]:
                count +=  w[filename]
            else:
                count -=  w[filename]
    if(count > 0):
        return 1
    return -1


# In[1]:


def condorcet_weighted (filename_list, path, dir_in, dir_out, dir_w="weights",nome_file="condorcetWeighted.txt"):
    global w
    w = findWeights(path, dir_in, dir_w)
    global condorcet
    global filename_list_global
    condorcet = {}  
    L = set({})
    for filename_in in filename_list:
        path_in = path + "\\" + dir_in + "\\" + filename_in
        in_file = pd.read_csv(path_in, delimiter = " ", header = None)
        in_file.columns = ["topicID", "q0", "docID", "rank", "score", "model"]
        
        for i in range(0, in_file.shape[0], 1):
                topicID = in_file['topicID'][i]
                documentID = in_file['docID'][i]
                score = in_file['score'][i]
                condorcet.setdefault((topicID, documentID), {})
                condorcet[(topicID, documentID)][filename_in] = score
                L.add((topicID,documentID))        
    
    filename_list_global = filename_list
    L = list(L) 
    LL = {}
    for i in L:
        LL.setdefault(i[0], [])
        LL[i[0]].append((i[0], i[1]))
        
    for k in LL:
        LL[k] = sorted(LL[k], cmp = algorithm1_weighted, reverse = True)
        LL[k] = [ np.asarray(list(LL[k][i]) + [i]) for i in range(len(LL[k]))]
        
        LL[k] = np.asarray(LL[k])
       
    Matrix=[]
    for k in LL:
        for i in LL[k]:
            Matrix.append(i)
    Matrix=np.asarray(Matrix) 
        
    path_out = path + "\\" + dir_out + "\\" + nome_file
    df = pd.DataFrame(data = Matrix, columns = ['topicID', 'docID', 'rank'])
    df['rank'] = df['rank'].astype('int64')
    df['score'] = df['rank'].max() - df['rank']  #1. / (df['rank'].astype('int64') + 1)
    df = df.sort_values(['topicID', 'rank'], ascending=[True, True])
    df['Q0'] = "Q0"
    df['model'] = nome_file
    df = df.reindex(columns = ['topicID', 'Q0', 'docID', 'rank', 'score', 'model'])
    df.to_csv(path_out, index = False, header = False, sep = " ") 
    
def condorcet_weightedML (filename_list, path, dir_in, dir_out, dir_w="weightsML",nome_file="condorcetWeightedML.txt"):
    condorcet_weighted (filename_list, path, dir_in, dir_out, dir_w,nome_file)
def condorcet_weightedLog (filename_list, path, dir_in, dir_out, dir_w="weightsLog",nome_file="condorcetWeightedLog.txt"):
    condorcet_weighted (filename_list, path, dir_in, dir_out, dir_w,nome_file)    


# # Utilities

# In[14]:


def listFiles(path, directory):
    filename_list = []
    for file in os.listdir(path + "\\" + directory):
        if(os.path.isfile(os.path.join(path + "\\" + directory, file))):
            filename_list.append(file)
    return filename_list

def manageDirectory(path, directory):
    if os.path.exists(path + "\\" + directory):
        os.chmod(path + "\\" + directory, 0777)
        #os.remove(path + "\\" + directory)
        shutil.rmtree(path + "\\" + directory)
    os.mkdir(path + "\\" + directory)    
    
def findWeights(path, directory, directory_weights,exist=True):
    filename_list = listFiles(path, directory)
    weights={}
    for filename in filename_list:
        map_value=take_MAP(path,directory,directory_weights,filename,exist)
        weights[filename]=map_value
    return weights

def take_MAP_TrecEval(path,directory, directory_weights,filename,exist=False,path_terrier=None):
    if(path_terrier==None):
        path_terrier=path
    if(not exist):
        process = "{}trec_eval {} {}"
        path_to_bin = path_terrier+"\\"+"terrier-core-4.2\\bin\\"
        path_to_pool = path_terrier+"\\"+"terrier-core-4.2\\share\\TIPSTER\\pool\\qrels.trec7.txt"
        path_to_run = path + "\\" + directory + "\\" + str(filename)
        process=str(process.format(path_to_bin, path_to_pool, path_to_run))
        #print("take MAP \n "+process)
        p = subprocess.check_output(process, shell=True)

        text_file = open(path+"\\" + directory_weights + "\\" + "w_"+ filename, "w")
        text_file.write(p)
        text_file.close()
        
    text_file = open(path+"\\" + directory_weights + "\\" + "w_"+ filename, "r")
    p=text_file.read()
     
    #print p
    lines=p.split("\n")        
    return float(lines[6].split()[2])

def take_MAP(path,directory, directory_weights,filename,exist=False,path_terrier=None):
    if(path_terrier==None):
        path_terrier=path    
    if(not exist):
        eng.addpath("r"+path, nargout=0)
        path_to_run = path + "\\" + directory + "\\" + str(filename)
        path_to_pool = path_terrier+"\\"+"terrier-core-4.2\\share\\TIPSTER\\pool\\qrels.trec7.txt"
        name_run=filename
        eng.workspace['path_to_pool']=path_to_pool
        eng.workspace['path_to_run']=path_to_run
        eng.workspace['name_run']=name_run
        eng.RankFusion(nargout=0)
        somma=eng.workspace['sum']

        text_file = open(path+"\\" + directory_weights + "\\" + "w_"+ filename, "w")
        text_file.write(str(somma))
        text_file.close()
        
    text_file = open(path+"\\" + directory_weights + "\\" + "w_"+ filename, "r")
    p=text_file.read()
       
    return float(p)


# # Test

# In[63]:


def main():
    # Could use Empty Line below if dirs are in the same python's working dir
    path = os.getcwd()
    
    dir_in = "input"   
    #filename_list = listFiles(path, dir_in)
    
    dir_w = "weights"   
    
    dir_norm = "norm"
    #If we already have normalized files than we can comment the next two rows
    #manageDirectory(path, dir_norm)
    #normalize_score_all(filename_list, path, dir_in, dir_norm)
    
    filename_list = listFiles(path, dir_norm)
    #If we want execute only some of the algorithm
    #we can comment line of manageDirectory and all the call to function of algorthims that we don't want
    dir_comb = "comb"
    #manageDirectory(path, dir_comb)
	
    #comb_sum(filename_list, path, dir_norm, dir_comb)
    #print "CombSum terminated without errors"
    #comb_max(filename_list, path, dir_norm, dir_comb)
    #print "CombMax terminated without errors"
    #comb_min(filename_list, path, dir_norm, dir_comb)
    #print "CombMin terminated without errors"
    #comb_median(filename_list, path, dir_norm, dir_comb)
    #print "CombMedian terminated without errors"
    #comb_mnz(filename_list, path, dir_norm, dir_comb)
    #print "CombMnz terminated without errors"
    #comb_anz(filename_list, path, dir_norm, dir_comb)
    #print "CombAnz terminated without errors"
    #condorcet_alg(filename_list, path, dir_norm, dir_comb)
    #print "Condorcet terminated without errors"
    #condorcet_weighted(filename_list, path, dir_norm, dir_comb, dir_w)
    #print "Condorcet Weighted terminated without errors"	
    condorcet_weightedML(filename_list, path, dir_norm, dir_comb)
    print "Condorcet WeightedML terminated without errors"
    condorcet_weightedLog(filename_list, path, dir_norm, dir_comb)
    print "Condorcet WeightedLog terminated without errors"		



