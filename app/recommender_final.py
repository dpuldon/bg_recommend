import MySQLdb as mdb #remember to source ~/.profile
import pandas as pd
import numpy as np
import scipy as scipy
import scipy.io
from scipy.optimize import minimize
from scipy.sparse import coo_matrix,csr_matrix,lil_matrix
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
from memory_profiler import memory_usage


def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def getRatingMatrix(end_userid,min_owners,min_ratings):

    debug = False
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    query = "SELECT a.id,b.colIndex,c.rating FROM usernames a,games_newnew b,user_game_relation c WHERE c.username = a.username AND c.game_id = b.id AND c.rating IS NOT NULL AND b.users_rated>'%s' AND b.users_owned>'%s' AND a.id<'%s'" % (min_ratings,min_owners,end_userid) # Note that the game has to be in the games table for this to work
    
    dataALC = pd.read_sql(query,con)
    datanumpy = np.array(dataALC,dtype=float)

    print "Number of non-zero entries = ",np.count_nonzero(datanumpy)
    print datanumpy.shape
    
    #spmat = coo_matrix(np.array([[  0. ,  0. ,  0.,   0.,   0.,   0. ,  0.,   0. ,  0. ,  0. ,  0.],[  0. ,  0. , 10. ,  8. , 10. ,  0. ,  0. , 10. ,  0. ,  7.  , 0.],[  7. ,  0. ,  8. ,  6. ,  8. ,  5. ,  0. ,  0. ,  1. ,  0. ,  4.],[  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , 10. ,  0. ,  0.],[ 10. ,  0. ,  7. ,  0. ,  7. ,  4. ,  0. , 10. ,  8. ,  5. ,  0.],[  0. ,  0. ,  6. ,  4. ,  6. ,  0. ,  9. ,  8. ,  4. ,  8. ,  5.],[  0. ,  0. ,  4. ,  9. ,  4. ,  9. ,  0. ,  9. ,  3. ,  3. ,  0.],[  8. ,  0. ,  0. ,  3. ,  9. ,  8. ,  0. ,  0. ,  7. ,  0. ,  0.]])) #Test Case, worked out in ipython notebook
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
        print "Number of User Reviews", num_rev
        print "Sum of User Reviews",sum_rev
        print "User Mean", User_mean

    rateM = coo_matrix((rateM_unNorm.tocoo().data-User_mean[Rcoords[1]],Rcoords),rateM_unNorm.shape).tocsr()   
    rateM = rateM.todense()

    if debug:
        print "Normalized Utility Matrix"
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
        print "Similarity Matrix"
        print cosim
        print cosim.shape

    print "Cosine Similarity Matrix Shape"
    print cosim.shape

    
    #Remove old pickle files
    files = glob.glob('/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/*.pkl')
    for f in files:
        os.remove(f) # Be careful about this
    
    for idx,i in enumerate(cosim):
        pickle.dump(i, open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/rating_cosine_similarity'+'{0}.pkl'.format(idx),"wb"))
        test = pickle.load(open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/rating_cosine_similarity'+'{0}.pkl'.format(idx),"rb"))
        if idx%1000==0:
            print "Now finishing this cosine similarity row: ",idx 
        if (test!=i).all():
            print "This pickle gives a different result:", i

def getOwnerMatrix(end_userid,min_owners,min_ratings):

    debug = False
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    query = "SELECT a.id,b.colIndex,c.owned FROM usernames a,games_newnew b,user_game_relation c WHERE c.username = a.username AND c.game_id = b.id AND c.owned=1 AND b.users_rated>'%s' AND b.users_owned>'%s' AND a.id<'%s'" % (min_ratings,min_owners,end_userid) # Note that the game has to be in the games table for this to work

    dataALC = pd.read_sql(query,con)


    datanumpy = np.array(dataALC,dtype=float)

    print datanumpy.shape

    spmat = coo_matrix((datanumpy.T[2],(datanumpy.T[1],datanumpy.T[0]))) #puts the shorter array, boardgames, in rows for faster computation
    rateM_unNorm = csr_matrix(spmat.todense(),dtype=float) #Utility matrix containing the user's game ownership from the query

    rateM_bool = csr_matrix(csr_matrix(rateM_unNorm,dtype=bool),dtype=float)
    
    rateM = rateM_unNorm.todense()

    if debug:
        print "Utility Matrix"
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

    print "Cosine Similarity Matrix Shape"
    print cosim.shape

    #rm old files
    files = glob.glob('/Users/davidpuldon/Projects/BGrecommend/Data/PickleOwnerFiles/*.pkl')
    for f in files:
        os.remove(f) # be careful about this
    
    for idx,i in enumerate(cosim):
        pickle.dump(i, open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleOwnerFiles/owner_cosine_similarity'+'{0}.pkl'.format(idx),"wb"))
        test = pickle.load(open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleOwnerFiles/owner_cosine_similarity'+'{0}.pkl'.format(idx),"rb"))
        if idx%1000==0:
            print "Now finishing this cosine similarity row: ",idx 
        if (test!=i).all():
            print "This pickle gives a different result:", i
    
def getPredictions(userlist,k_thres,nPlayers,playtime,hideratedgames):

    debug = True

    if nPlayers == -1:
        minPlayers = 999
        maxPlayers = 0
    else:
        minPlayers = nPlayers
        maxPlayers = nPlayers

    if playtime == -1:
        playtime = 9999
        
    if '&' in userlist:
        g=userlist.replace("&"," ")
        userlist =  g.split()
    
    user_ratings = []
    user_ownership = []
    
    #loops over user ids and returns their ratings and owner ship
    for user in userlist:
        (ratingstats,ownerstats)=parse_collection(user)
        user_ratings.append(ratingstats)
        user_ownership.append(ownerstats)

    #Makes a Composite 'person' out of the inputted userlist
    if len(userlist)>1:
        (group_ratings,group_ownership)=make_group(user_ratings,user_ownership)
    else:
        (group_ratings,group_ownership) = (user_ratings[0],user_ownership[0])

    
    path= '/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/'
    # simple version for working with CWD
    gamelist_tot = len([name for name in os.listdir(path) if name.endswith('.pkl')])
    
    rating_cosine_similarity = csr_matrix((gamelist_tot,gamelist_tot),dtype=np.float).toarray()
    owner_cosine_similarity = csr_matrix((gamelist_tot,gamelist_tot),dtype=np.float).toarray()
    for i in group_ratings:
        tmp_pkl = pickle.load(open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/rating_cosine_similarity'+'{0}.pkl'.format(i[0]),"rb"))
        rating_cosine_similarity[:,i[0]]= tmp_pkl
        rating_cosine_similarity[i[0],:]= tmp_pkl

    for i in group_ownership:
        tmp_pkl = pickle.load(open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleOwnerFiles/owner_cosine_similarity'+'{0}.pkl'.format(i[0]),"rb"))
        owner_cosine_similarity[:,i[0]]= tmp_pkl
        owner_cosine_similarity[i[0],:]= tmp_pkl

    
    
    #Restricts the games to take into acount for the prediction based off if they are greater than the similiarity threshold set by k_thres
    if k_thres > -1:
        rating_cosim = threshold(rating_cosim,k_thres)
    
    group_rating_vector = np.zeros(gamelist_tot)
    group_owner_vector = np.zeros(gamelist_tot)
    
    for rating_pair in group_ratings:
        group_rating_vector[rating_pair[0]]=rating_pair[1]
    for owner_pair in group_ownership:
        group_owner_vector[owner_pair[0]]=owner_pair[1]

    group_rating_vector = csr_matrix(group_rating_vector)
    group_owner_vector = csr_matrix(group_owner_vector)
    print group_rating_vector
    
    #owner predictions
    owner_pred = group_owner_vector.dot(owner_cosine_similarity)
    alr_owned = np.array(np.logical_not(np.array(group_owner_vector.todense()).flatten()),dtype=float).flatten()
    #remove games already owned
    owner_pred = np.array(owner_pred).flatten()*alr_owned
    print "owner predictions shape: ", owner_pred.shape
    
    group_rating_bool_vector= csr_matrix(csr_matrix(group_rating_vector,dtype=bool),dtype=float)
    alr_rated = np.array(np.logical_not(group_rating_bool_vector.todense()),dtype=float).flatten()
    
    pred_part1= group_rating_vector.dot(rating_cosine_similarity)

    pred_part2 = group_rating_bool_vector.dot(np.fabs(rating_cosine_similarity))

    rating_predictions = np.nan_to_num(pred_part1/pred_part2)

    rating_predictions = np.array(rating_predictions).flatten()
    
    for rating_pair in group_ratings:
        rating_predictions[rating_pair[0]] = rating_pair[1] #just replaces prediction for actual user rating, for items already rated
        
    if hideratedgames is True:
        rating_predictions = rating_predictions*alr_rated

    rating_predictions = rating_predictions*alr_owned # removes games you own
    
    print "rating predictions shape: ",rating_predictions.shape
    
    #This is where I develop two queries, one using just rating information, the other using rating owner information

    #need to turn predictions into a dict with the game index so I can sort and then look up
    dict_ratings = dict(enumerate(rating_predictions))
    dict_ratings_sort = sorted(dict_ratings.items(),key=itemgetter(1))
    dict_ratings_reverse = dict_ratings_sort[::-1]

    if debug:
        print "Worst ratings"
        print dict_ratings_sort[:10]
        print "Rating Predictions!!"
        print dict_ratings_reverse[:10]

    query_result = dict_ratings_reverse[:1000]

    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket
    cur = con.cursor()

    #Define how many ranks you want to sort by the owner information
    nRanks_organized = 50
    
    predQuery = []
    rating_rank = {}
    for idx,query in enumerate(query_result):
        cur.execute("""SELECT name,mechanics,id,thumbnail,min_players,max_players,playtime,year FROM games_newnew WHERE colindex = '%s' AND playtime<='%s' AND min_players <= '%s' AND max_players >= '%s'""" % (query[0],playtime,minPlayers,maxPlayers))
        tmpQuery = cur.fetchall()
        if tmpQuery == ():
            print "row is empty, skipping index"
            continue
        predQuery.append(tmpQuery[0])
        rating_rank[query[0]]=idx
        if len(predQuery)==nRanks_organized:
            break
    if debug:   
        print "Rating Ranks"
        print rating_rank

    #Get the purchase similarity for the rating ranked k games
    owned_values = {}
    for game_id in rating_rank.keys():
        owned_values[game_id]= owner_pred[game_id]

    dict_owner_sort = sorted(owned_values.items(),key=itemgetter(1))
    dict_owner_reverse = dict_owner_sort[::-1]
    if debug:
        print owned_values
        print " "
        print dict_owner_reverse
    owner_rank = {}
    total_rank = {}
    for idx,game in enumerate(dict_owner_reverse):
        owner_rank[game[0]]=idx
        total_rank[game[0]]=idx + rating_rank[game[0]]

    if debug:
        print "Owner Ranks"
        print owner_rank
        print "Total Ranks"
        print total_rank
        
    dict_total_sort = sorted(total_rank.items(),key=itemgetter(1))

    if debug:
        print "total sorted ranks"
        print dict_total_sort

    predTotalQuery = []
    for idx,query in enumerate(dict_total_sort):
        cur.execute("""SELECT name,mechanics,id,thumbnail,min_players,max_players,playtime,year FROM games_newnew WHERE colindex = '%s' AND playtime<='%s' AND min_players <= '%s' AND max_players >= '%s'""" % (query[0],playtime,minPlayers,maxPlayers))
        tmpQuery = cur.fetchall()
        if tmpQuery == ():
            print "row is empty, skipping index"
            continue
        predTotalQuery.append(tmpQuery[0])
        rating_rank[query[0]]=idx
        if len(predTotalQuery)==nRanks_organized:
            break
    
    if debug:
        for query in predQuery[:20]:
            print query

        print "\n Rating+Owner query results sorted \n"
        for query in predTotalQuery[:20]:
            print query

    return (predQuery[:12],predTotalQuery[:12])
    

def make_group(user_ratings,user_ownership):
    #turns a list of users into 1 person for predictions

    rating_dic = {}
    owner_dic = {}
    for user in user_ratings:
        for game in user:
            if game[0] in rating_dic:
                rating_dic[game[0]].append(game[1])
            else:
                rating_dic[game[0]]=list()
                rating_dic[game[0]]=[game[1]]

    for key in rating_dic:
        rating_dic[key] = np.mean(rating_dic[key])

    for user in user_ownership:
        for game in user:
            if game[0] in owner_dic:
                owner_dic[game[0]].append(game[1])
            else:
                owner_dic[game[0]]=list()
                owner_dic[game[0]]=[game[1]]

    
    for key in owner_dic:
        owner_dic[key] = np.mean(owner_dic[key])
    
    return (rating_dic.items(),owner_dic.items())

def get_collection(USER):

    # return collection for any user, but wait some time and retry if error. 20 total attempts.
    
    url = 'http://www.boardgamegeek.com/xmlapi2/collection?username='

    for i in range(20): # try 20 times
	r = requests.get(url + USER + "&stats=1")
	if r.status_code == 202:
	    time.sleep(math.pow(i, 1.5) )
	    i += 1
	    continue
	else:
	    #print '# of query attempts is {}'.format(i+1)
	    data = r.content
	    return data
            break


def parse_collection(user):

    ownerstats = []
    ratingstats = []
    
    coll = get_collection(user)
    parsed = etree.fromstring(coll)
    ids = [x.items()[1][1] for x in parsed]
    
    ratingValue = [rating.attrib for rating in parsed.iter('rating')]
    ownStatus=[own.attrib for own in parsed.iter('status')]
    #wantStatus=[want.attrib for want in parsed.iter('status')]
    #wanttoplayStatus=[wanttoplay.attrib for wanttoplay in parsed.iter('status')]
    #wanttobuyStatus=[wanttobuy.attrib for wanttobuy in parsed.iter('status')]
    #prevownedStatus=[prevowned.attrib for prevowned in parsed.iter('status')]
    #wishlistStatus=[wishlist.attrib for wishlist in parsed.iter('status')]
    #fortradeStatus=[fortrade.attrib for fortrade in parsed.iter('status')]
    gameName = [item.find('name').text for item in parsed.iter('item')]


    duplicates = sorted(list_duplicates(ids))
    removeIndices = []
    for dup in duplicates:
        removeIndices.append(dup[1][:-1])

    skip_indices = list(chain.from_iterable(removeIndices))

    for i,parse in enumerate(parsed):

        if i in skip_indices:
            continue

        #check if game id is in the games database
        con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket
        with con:
            cur = con.cursor()
            cur.execute("""SELECT games_newnew.id,games_newnew.colIndex FROM games_newnew WHERE ID = '%s'  AND expansion = 0;""" %(ids[i]))
            rows = cur.fetchall()

        #print "this is rows for id: ", ids[i], rows
        if rows == ():
            #print "row is empty, skipping index"
            continue
        
        
        if ratingValue[i]['value'] != 'N/A':
            item_rating = float(ratingValue[i]['value'])
        else:
            item_rating = None

        
        own = int(ownStatus[i]['own'])

        #passes the colIndex for each game to make the similarity matrix look up easier
        if item_rating is not None:
            ratingstats.append((int(rows[0][1]),item_rating))

        if own ==1:
            ownerstats.append((int(rows[0][1]),1))
    
            
        
    return (ratingstats,ownerstats)

def doValidation(end_userid,k_thres):
    #This should loop through a fraction of the users not used by the training set to make the similarity matrix
    #From this set, loop through each user and loop through their ratings, hiding 1 item and then getting the rating prediction for the hidden item
    #do this for each item per user
    #things to return, # of ratings per user (minus the holdout), P-R per item(a dict or df)
    
    Preddiff = []
    
    #Get a list of usernames for ids larger than the end_userlist
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    cur = con.cursor()
    cur.execute("""SELECT username FROM usernames WHERE id>'%s';""" % (end_userid))
    
    data = cur.fetchall()

    userlist = []
    for user in data:
        userlist.append(user[0])

    #For testing
    #userlist = ['dpuldon']
    #userlist = ['dpuldon','msaya']
    #userlist = ['msaya']
    #print "Validation user list: ",userlist

    #unpack the entire similarity matrix here so it isnt in the loop
    path= '/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/'
    # simple version for working with CWD
    gamelist_tot = len([name for name in os.listdir(path) if name.endswith('.pkl')])
    
    rating_cosim = np.zeros((gamelist_tot,gamelist_tot))
    for i in xrange(gamelist_tot):
        tmp_pkl = pickle.load(open('/Users/davidpuldon/Projects/BGrecommend/Data/PickleFiles/rating_cosine_similarity'+'{0}.pkl'.format(i),"rb"))
        rating_cosim[i,:]= tmp_pkl
        rating_cosim[:,i]= tmp_pkl
        

    if k_thres > -1:
        rating_cosim = threshold(rating_cosim,k_thres)

    rating_cosine_similarity = csr_matrix(rating_cosim)
    
    #loops over user ids and returns their ratings and owner ship
    user_ratings = []
    user_ownership = []
    ErrorDict = {}
    for u_idx,user in enumerate(userlist):
        if u_idx%50==0:
            print "On validation user: ",u_idx
        #(ratingstats,ownerstats)=parse_collection(user)
        cur.execute("""SELECT b.colIndex, c.rating FROM games_newnew b,user_game_relation c WHERE b.id=c.game_id AND c.rating IS NOT NULL AND c.username='%s';""" % (user))
        data = cur.fetchall()
        ratingstats =[]
        for rating in data:
            ratingstats.append(rating)
        
        #print "rating stats",ratingstats
        #print user_ratings,'\n'
        #print user_ownership
        nRating = len(ratingstats)
        
        for idx,rating in enumerate(ratingstats):
            tmpratingstats = copy.deepcopy(ratingstats)
            hidden_gameid = rating[0]
            hidden_rating = rating[1]
            #print hidden_gameid,hidden_rating
            del tmpratingstats[idx] #delete the hidden rating from the original user's rating

            group_rating_vector = np.zeros(len(rating_cosim))
            for rating_pair in tmpratingstats:
                group_rating_vector[rating_pair[0]]=rating_pair[1]

            
            group_rating_vector = csr_matrix(group_rating_vector)

            group_rating_bool_vector= csr_matrix(csr_matrix(group_rating_vector,dtype=bool),dtype=float)
            alr_rated = np.array(np.logical_not(group_rating_bool_vector.todense()),dtype=float).flatten()
            pred_part1= group_rating_vector.dot(rating_cosine_similarity)
            pred_part2 = group_rating_bool_vector.dot(np.absolute(rating_cosine_similarity))
            
            rating_predictions = np.nan_to_num(pred_part1/pred_part2)

            rating_predictions = np.array(rating_predictions).flatten()

            rating_predictions = rating_predictions*alr_rated
            
            #print "hidden rating: ",hidden_rating,",predicted rating = ",rating_predictions[hidden_gameid],", P-R = ",hidden_rating-rating_predictions[hidden_gameid]
            Preddiff.append(hidden_rating-rating_predictions[hidden_gameid])

            if ErrorDict.has_key(nRating-1):
                ErrorDict[nRating-1][0]=ErrorDict[nRating-1][0]+1 # add to average count
                ErrorDict[nRating-1][1]=ErrorDict[nRating-1][1]+abs(hidden_rating-rating_predictions[hidden_gameid]) # add Pred-Rating to sum
                ErrorDict[nRating-1][2]=ErrorDict[nRating-1][2]+(hidden_rating-rating_predictions[hidden_gameid])**2 # add squared Pre-Rating to sum
            else:
                ErrorDict[nRating-1]=[1,abs(hidden_rating-rating_predictions[hidden_gameid]),(hidden_rating-rating_predictions[hidden_gameid])**2]

    #print ErrorDict
    pickle.dump(ErrorDict, open("/Users/davidpuldon/Projects/BGrecommend/app/Validation_error_dict.pkl","wb"))
    
    testpkl = pickle.load(open("/Users/davidpuldon/Projects/BGrecommend/app/Validation_error_dict.pkl","rb"))

    if ErrorDict == testpkl:
        print "Pickling worked!"

    
    print "The average difference b/w Prediction and User Rating is = ",np.mean(np.absolute(Preddiff))

    return
    

    
if __name__ == "__main__":

    end_userid = 53405
    min_owners = 50
    min_ratings=200
    #getRatingMatrix(end_userid,min_owners,min_ratings)

    #getOwnerMatrix(end_userid,min_owners,min_ratings)

    #t0 = time.time()
    #doValidation(end_userid,0)
    #t1 = time.time()
    #print "total time for predictions = ", t1-t0

    #Test Usernames for getPredictions
    userlist = ['dpuldon']
    #userlist = ['dpuldon','dpuldon2']
    #userlist = ['dpuldon','dpuldon2','msaya']
    #userlist = ['msaya']

    nPlayers = -1
    playtime = -1
    hideratedgames = True
    #t0 = time.time()
    getPredictions(userlist,-1,nPlayers,playtime,hideratedgames) #gives prediction
    #t1 = time.time()
    #print "total time for predictions = ", t1-t0

    #mem_usage = memory_usage(blah_test)
    #print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    #print('Maximum memory usage: %s' % max(mem_usage))
    
