import MySQLdb as mdb #remember to source ~/.profile
#import pymysql as mdb
import pandas as pd
import os


def create_userlist():
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket


    with con:
        cur = con.cursor()
        cur.execute("SELECT User_id FROM users GROUP BY User_id HAVING Count(Rating)>20")
        rows = cur.fetchall()
        print len(rows)

    return(rows)

if __name__ == "__main__":
    user_tuples = create_userlist()
    print len(user_tuples)
    users = [user[0] for user in user_tuples]
    print len(users)
    print users[0],users[400],users[50000]
