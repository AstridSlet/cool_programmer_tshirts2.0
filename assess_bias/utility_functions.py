#Import  packages
import numpy as np
import gensim
#import danlp
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import sys, os
sys.path.append(os.path.join('..'))

def cosine_sim(v1, v2, embedding):
    """
    Returns cosine of the angle between two vectors
    The function first loads the embeddings for the two words v1 and v2 that the cosine similarity has to be calculated for. 
    It then calculates the cosine similarity using the formula: cos(x, y) = x . y / |x| * |y|
    np.dot returns the dot product of the vectors x and y (x . y)
    np.linalg.norm returns the length of the given vector (|x| and |y|)
    
    """
    v1 = embedding[v1]
    v2 = embedding[v2]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos

def weat_association(W, A, B, embedding):
    """
    The function makes two lists - one that stores the cosine distance between a target word W and all of attribute words in A and one storing the cosine distance between the target word W and all attribute words in B.
    Cosine distance is calculated by the function defined as cosine_sim 
    The function returns the difference in means between the word W's association with the attributes A and B
    Returns what corresponds to s(w, A, B) in the equations defined in the thesis

    """
    cosine_scoresA = [] #create an empty list for associations between W and words in A
    cosine_scoresB = [] #create an empty list for associations between W and words in B
    for i in A:
        cosine_scoresA.append(cosine_sim(W, i, embedding))
    for i in B:
        cosine_scoresB.append(cosine_sim(W, i, embedding))
        
    association = np.mean(cosine_scoresA) - np.mean(cosine_scoresB)
    
    return association

def weat_differential_association(X, Y, A, B, embedding):
    """
    The function calculates the differential association
    It uses the function defined above (weat_association) to calculate the different in association between a given 
    word W's association with words in respectively A and B - but it iterates for all words W in respectively X and Y
    Returns what corresponds to s(X, Y, A, B) in the equations
    """
    associationX = []
    associationY = []
    for i in X:
        associationX.append(weat_association(i, A, B, embedding))
    for i in Y:
        associationY.append(weat_association(i, A, B, embedding))
    
    diff_association = np.sum(associationX) - np.sum(associationY)
    
    return diff_association

def weat_effect_size(X, Y, A, B,embedding):
    """
    Calculates the WEAT effect size as described in the thesis
    """
      
    associationX = []
    associationY = []
    for i in X:
        associationX.append(weat_association(i, A, B, embedding))
    for i in Y:
        associationY.append(weat_association(i, A, B, embedding))
    associationXY = []
    associationXY = associationX + associationY
    tmp1 = np.mean(associationX) - np.mean(associationY)
    tmp2 = np.std(associationXY,ddof=1) 
        
    effect_size = tmp1/tmp2
    
    return effect_size

#Permutation test p-value - has to be adjusted + code has to run more smoothly
def weat_p_value(X, Y, A, B, embedding, p):
    """
    Returns one-sided p-value of the permutation test 
    What the permutation test basically does: makes a lot (i) possible combinations of our target words and assesses
    their association to the attributes. The proportion of differential association that are higher for permuted distributions 
    than for the non permuted distrubition are then calculated, which is the permutation test p-value

    """
    diff_association = weat_differential_association(X, Y, A, B, embedding)
    target_words = np.concatenate((X, Y), axis=0)
    np.random.shuffle(target_words) #shuffle target words before permutations
    
    #Test if target words can be divided into two sets of equal size - otherwise print warning
    if target_words.shape[0] % 2 != 0:
        print('WARNING - target word set can not be divided into two sets of equal size')
        
    partition_diff_association = [] #Create empty list to be filled during loop

    for i in range(p): #Iterate p times (number of permutations)
        seq = np.random.permutation(target_words) #Permute target words
        partition_X = seq[:len(seq)//2] #Load  first partition of data to create to sets of permuted target words
        partition_Y = seq[len(seq)//2:] #Load second partition of data to create to sets of permuted target words
        #Calculate and append differential association for permuted target words to attributes
        partition_diff_association.append(weat_differential_association(partition_X, partition_Y, A, B, embedding))
      
    partition_diff_association = np.array(partition_diff_association) #Convert differential association for all permuted samples to numpy array

    mean = np.mean(partition_diff_association) #Mean differential association for permutations
    stdev = np.std(partition_diff_association) #Standard deviation of differential association for permutations
    pvalue = ((np.sum(partition_diff_association > diff_association)+1) / (len(partition_diff_association)+1)) #Calculation of p-value, corresponds to proportion of differential association for permuted target words that are higher than for the non permuted value
    
    return diff_association, mean, stdev, pvalue

#First we make one function that returns all relevant values for one embedding and one kind of gender bias


#The next section runs the results for each model individually, 
#as it is too computationally expensive to run all models together

def results_weat(X, Y, A, B, embedding, p):
    #Define empty lists
    diff_association = []
    effect_size = []
    pvalue = []

    #Retrieve values from already defined functions
    diff_association = weat_differential_association(X, Y, A, B, embedding)
    effect_size = weat_effect_size(X, Y, A, B,embedding)
    pvalue = weat_p_value(X, Y, A, B, embedding, p)[3]
    
    #Combine all values in dataframe
    s1=pd.Series(diff_association,name='diff_association')
    s2=pd.Series(effect_size,name='effect_size')
    s3=pd.Series(pvalue ,name='pvalue')

    results = pd.concat([s1,s2,s3], axis=1)
        
    #Return dataframe with all results
    return results

#Now run the tests for all types of gender biases (science vs. arts, math vs. arts, career vs. family) 
# + all pre-trained models (separately as computational strain is too big otherwise)

def weat_func(wordembedding, model_name, theme1, theme2, permutations, male, female, word_list1, word_list2):
    print("running")
    # set permutations
    p = permutations

    # set target and attribute words - Career vs. Family
    sub1 = word_list1
    sub2 = word_list2
    
    # run WEAT
    results = results_weat(sub1, sub2, male, female, wordembedding, p)

    # get name
    name = f"WEAT_{theme1}_{theme2}_{model_name}.csv"

    # save results 
    results.to_csv(os.path.join("/work/Exam/dk-weat/output/", name), index = True)

    return print(results)

