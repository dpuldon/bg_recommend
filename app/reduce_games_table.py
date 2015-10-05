import MySQLdb as mdb #remember to source ~/.profile
import pandas as pd
import numpy as np
import scipy as scipy
import scipy.io
from scipy.optimize import minimize
from scipy.sparse import coo_matrix,csr_matrix
from scipy.stats import threshold
import os,os.path
import requests
from lxml import etree
from itertools import chain
from collections import defaultdict
from operator import itemgetter
import time
import math
import pickle
import time
import glob
import copy

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def create_reduced_games_table(end_userid,min_owners,min_ratings):

    debug = False
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    query = "SELECT a.id,b.colIndex,c.rating FROM usernames a,games_new b,user_game_relation c WHERE c.username = a.username AND c.game_id = b.id AND c.rating IS NOT NULL AND b.users_rated>'%s' AND b.users_owned>'%s' AND a.id<'%s'" % (min_ratings,min_owners,end_userid) # Note that the game has to be in the games table for this to work

    dataALC = pd.read_sql(query,con)

    datanumpy = np.array(dataALC,dtype=float)

    print "Number of non-zero entries = ",np.count_nonzero(datanumpy)
    #return

    #print datanumpy
    print datanumpy.shape
    

    spmat = coo_matrix((datanumpy.T[2],(datanumpy.T[1],datanumpy.T[0]))) #puts the shorter array, boardgames, in rows for faster computation
    rateM_unNorm = csr_matrix(spmat.todense(),dtype=float) #Utility matrix containing the user ratings from the query

    rateM_bool = csr_matrix(csr_matrix(rateM_unNorm,dtype=bool),dtype=float)

    Rcoo = rateM_bool.tocoo()
    Rcoords = (Rcoo.row,Rcoo.col)

    if debug:
        print "debugging"
        print rateM_unNorm
        print "boolean"
        print rateM_bool
    
    sum_rev = np.array(rateM_unNorm.sum(axis=0)).flatten() 
    num_rev = np.array(rateM_bool.sum(axis=0)).flatten()
    User_mean=np.nan_to_num(sum_rev/num_rev)

    if debug:
        print "number of reviews and sum of reviews"
        print num_rev,sum_rev
        print "mean"
        print User_mean

    rateM = coo_matrix((rateM_unNorm.tocoo().data-User_mean[Rcoords[1]],Rcoords),rateM_unNorm.shape).tocsr()

    
    
    rateM = rateM.todense()

    if debug:
        print "im here"
        print rateM
    
    item_similarity = np.dot(rateM,rateM.T)

    square_mag = np.diag(item_similarity)

    inv_square_mag = 1/(square_mag)

    inv_square_mag[np.isinf(inv_square_mag)]=0

    inv_mag = np.sqrt(inv_square_mag)

    if debug:
        print "now here"
        print item_similarity
        print "yup"
        print inv_mag
        print item_similarity.shape,inv_mag.shape
        print item_similarity.ndim,inv_mag.ndim

    
    cosim = np.array(item_similarity)*inv_mag
    if debug:
        print "mid-test"
        print cosim
    cosim = cosim.T*inv_mag #cosine similarity for items (boardgames) 

    if debug:
        print cosim
        print cosim.shape

    count_zeros = 0
    for idx,sim_row in enumerate(cosim):
        if len(cosim)-np.count_nonzero(sim_row) == 11032:
            count_zeros +=1
            #print "Index: ",idx, "number of zeros = ",len(cosim)-np.count_nonzero(sim_row)
        else:
            with con:
                cur = con.cursor()
                cur.execute("INSERT into games_newnew (id,name,expansion,year,playtime,bggrank,image,thumbnail,min_players,max_players,min_age,rating_average,rating_average_weight,rating_bayes_average,rating_median,rating_num_weights,rating_stddev,users_commented,users_owned,users_rated,users_trading,users_wanting,users_wishing,mechanics,categories,description) SELECT a.id,a.name,a.expansion,a.year,a.playtime,a.bggrank,a.image,a.thumbnail,a.min_players,a.max_players,a.min_age,a.rating_average,a.rating_average_weight,a.rating_bayes_average,a.rating_median,a.rating_num_weights,a.rating_stddev,a.users_commented,a.users_owned,a.users_rated,a.users_trading,a.users_wanting,a.users_wishing,a.mechanics,a.categories,a.description FROM games_new a WHERE colIndex = '%s'" % (idx))
                      
            
    print "Number of rows with zero similarity",count_zeros
    
    return

def create_games_table():
    
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket


    with con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS games_newnew")
        cur.execute("CREATE TABLE games_newnew(colIndex INT AUTO_INCREMENT,id INT UNIQUE,name VARCHAR(100),expansion INT,year INT,playtime INT,bggrank INT,image VARCHAR(255),thumbnail VARCHAR(255),min_players INT,max_players INT,min_age INT,rating_average FLOAT,rating_average_weight FLOAT,rating_bayes_average FLOAT,rating_median FLOAT,rating_num_weights FLOAT,rating_stddev FLOAT,users_commented INT,users_owned INT,users_rated INT,users_trading INT, users_wanting INT,users_wishing INT,mechanics BLOB,categories BLOB,description BLOB,PRIMARY KEY(colIndex))")

if __name__ == "__main__":

    create_games_table()
    end_userid = 53405
    min_owners = 50
    min_ratings=200
    create_reduced_games_table(end_userid,min_owners,min_ratings) #Creates a games table where the games with rows of  all zeros in the similarity matrix are removed
