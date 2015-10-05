import MySQLdb as mdb #remember to source ~/.profile
import pandas as pd
import os
import requests
import time
import math
from lxml import etree
from collections import defaultdict
from itertools import chain


def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

def create_userlist():
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket


    with con:
        cur = con.cursor()
        cur.execute("SELECT User_id FROM users GROUP BY User_id HAVING Count(Rating)>20")
        rows = cur.fetchall()
        print len(rows)

    return(rows)

def get_usernames(file):
    with open(file,'r') as f:
        usernames = [line.strip() for line in f]

    return usernames

def create_username_sqltable():

    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS usernames")
    cur.execute("CREATE TABLE usernames(id INT AUTO_INCREMENT,username VARCHAR(50) NOT NULL UNIQUE,avg_rating FLOAT,total_owned INT,total_want INT,total_wanttoplay INT,total_wanttobuy INT,total_prevowned INT, total_wishlist INT,total_fortrade INT,PRIMARY KEY(id))")

    con.commit()
    con.close()

def create_usertogame_relational_table():

    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS user_game_relation")
    cur.execute("CREATE TABLE user_game_relation(transaction INT AUTO_INCREMENT UNIQUE,username VARCHAR(50) NOT NULL,game_id INT NOT NULL, rating FLOAT,owned INT, want INT, wanttoplay INT, wanttobuy INT, prevowned INT, wishlist INT, fortrade INT,PRIMARY KEY(username,game_id))")
    con.commit()
    con.close()

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
    
def fill_user_usertogame_tables(users):

    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    cur = con.cursor()
    tmpfile = open('users_done.txt','w')
    
    for idx,user in enumerate(users):
        coll = get_collection(user)

        parsed = etree.fromstring(coll)

        ids = [x.items()[1][1] for x in parsed]

        ratingValue = [rating.attrib for rating in parsed.iter('rating')]
        ownStatus=[own.attrib for own in parsed.iter('status')]
        wantStatus=[want.attrib for want in parsed.iter('status')]
        wanttoplayStatus=[wanttoplay.attrib for wanttoplay in parsed.iter('status')]
        wanttobuyStatus=[wanttobuy.attrib for wanttobuy in parsed.iter('status')]
        prevownedStatus=[prevowned.attrib for prevowned in parsed.iter('status')]
        wishlistStatus=[wishlist.attrib for wishlist in parsed.iter('status')]
        fortradeStatus=[fortrade.attrib for fortrade in parsed.iter('status')]
        gameName = [item.find('name').text for item in parsed.iter('item')]

        #Initialize composite totals
        User_avg_rating = 0
        User_total_rating = 0
        games_rated = 0
        total_owned = 0
        total_want =0
        total_wanttoplay = 0
        total_wanttobuy = 0
        total_prevowned = 0
        total_wishlist = 0
        total_fortrade = 0

        duplicates = sorted(list_duplicates(ids))
        removeIndices = []
        for dup in duplicates:
            removeIndices.append(dup[1][:-1])

        skip_indices = list(chain.from_iterable(removeIndices))

        for i,parse in enumerate(parsed):

            if i in skip_indices:
                continue
            '''
            print "Game Name: ",gameName[i]
            print "Game ID: ",ids[i]
            print "Game Rating: ",ratingValue[i]['value']
            print "Owner Status: ",ownStatus[i]['own']
            print "Want Status: ",wantStatus[i]['want']
            print "Want to play Status: ",wanttoplayStatus[i]['wanttoplay']
            print "Want to buy Status: ", wanttobuyStatus[i]['wanttobuy']
            print "Previously Owned Status: ",prevownedStatus[i]['prevowned']
            print "Wishlist Status: ", wishlistStatus[i]['wishlist']
            print "For Trade Status: ",fortradeStatus[i]['fortrade']+'\n'
            '''

            if ratingValue[i]['value'] != 'N/A':
                item_rating = float(ratingValue[i]['value'])
            else:
                item_rating = None
            
            # Fill the user to game relational table
            cur.execute('''INSERT into user_game_relation(username,game_id,rating,owned,want,wanttoplay,wanttobuy,prevowned,wishlist,fortrade)
                       values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                        (user,int(ids[i]),item_rating,int(ownStatus[i]['own']),int(wantStatus[i]['want']),int(wanttoplayStatus[i]['wanttoplay']),int(wanttobuyStatus[i]['wanttobuy']),int(prevownedStatus[i]['prevowned']),int(wishlistStatus[i]['wishlist']),int(fortradeStatus[i]['fortrade'])))

            

            
            if ratingValue[i]['value'] != 'N/A':
                #print "rating is true" +'\n'
                User_total_rating += float(ratingValue[i]['value'])
                games_rated += 1
        
            total_owned += int(ownStatus[i]['own'])
            total_want += int(wantStatus[i]['want'])
            total_wanttoplay += int(wanttoplayStatus[i]['wanttoplay'])
            total_wanttobuy += int(wanttobuyStatus[i]['wanttobuy'])
            total_prevowned += int(prevownedStatus[i]['prevowned'])
            total_wishlist += int(wishlistStatus[i]['wishlist'])
            total_fortrade += int(fortradeStatus[i]['fortrade'])
            
        try:    
            User_avg_rating = User_total_rating/float(games_rated)
        except ZeroDivisionError:
            User_avg_rating = None

        #Fill the user table
        cur.execute('''INSERT into usernames(username,avg_rating,total_owned,total_want,total_wanttoplay,total_wanttobuy,total_prevowned,total_wishlist,total_fortrade)
                    values(%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (user,User_avg_rating,total_owned,total_want,total_wanttoplay,total_wanttobuy,total_prevowned,total_wishlist,total_fortrade))

        con.commit()
        #Add user_name to new txt file to be able to restart later
        tmpfile.write(str(user)+'\n')

        if idx%10==0:
            print "Users Processed: ",idx

    con.close()
    tmpfile.close()
    
    
        
if __name__ == "__main__":
    ''' Just makes a new text file, so you can re-run this if you want to use users other than >20 ratings
    user_tuples = create_userlist() # Looks only at users with >20 ratings
    print len(user_tuples)
    users = [user[0].strip() for user in user_tuples]

    tmpfile = open('usernames_gt20_ratings.txt','w')
    for user in users:
        tmpfile.write(str(user)+'\n')
    tmpfile.close()
    '''

    #Initializes/RECREATES the username sql table which stores user information for each username
    ###create_username_sqltable() #Only do if you have a backup

    #Initializes/RECREATES the user-game relational "transaction" table
    ###create_usertogame_relational_table() #Only do if you have a backup

    user_list_file = '/Users/davidpuldon/Projects/BGrecommend/Data/usernames_gt20_ratings_mod.txt'
    username_list = get_usernames(user_list_file)
    #username_list = ['-Johnny-']#,'dpuldon','Zoids','Zealotrush','yonostudio'] # for testing

    #Fills the username and usertogame relational table using the username_list
    fill_user_usertogame_tables(username_list)

    
