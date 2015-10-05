import MySQLdb as mdb
import pandas as pd
from boardgamegeek import BoardGameGeek

def create_games_table():
    
    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket


    with con:
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS games_new")
        cur.execute("CREATE TABLE games_new(colIndex INT AUTO_INCREMENT,id INT UNIQUE,name VARCHAR(100),expansion INT,year INT,playtime INT,bggrank INT,image VARCHAR(255),thumbnail VARCHAR(255),min_players INT,max_players INT,min_age INT,rating_average FLOAT,rating_average_weight FLOAT,rating_bayes_average FLOAT,rating_median FLOAT,rating_num_weights FLOAT,rating_stddev FLOAT,users_commented INT,users_owned INT,users_rated INT,users_trading INT, users_wanting INT,users_wishing INT,mechanics BLOB,categories BLOB,description BLOB,PRIMARY KEY(colIndex))")


def get_game_ids(file):
    with open(file,'r') as f:
        game_ids = [line.strip() for line in f]

    return game_ids

def fill_game_table(game_ids):
    bgg = BoardGameGeek()

    con = mdb.connect(user= 'root', passwd='', db='userlist',unix_socket="/tmp/mysql.sock") #user, password, #database, #unix socket

    cur = con.cursor()
    tmpfile = open('gameIDs_done.txt','w')
    
    for idx,id in enumerate(game_ids):

        
        
        if idx%10==0:
            print "On file: ",idx

        
        g = bgg.game(game_id=id)
        
        try:
            cur.execute('''INSERT into games_new (id,name,expansion,year,playtime,bggrank,image,thumbnail,min_players,max_players,min_age,rating_average,rating_average_weight,rating_bayes_average,rating_median,rating_num_weights,rating_stddev,users_commented,users_owned,users_rated,users_trading,users_wanting,users_wishing,mechanics,categories,description)
                       values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
                        (g.id,g.name.encode('latin-1','ignore'),g.expansion,g.year,g.playing_time,g.boardgame_rank,g.image,g.thumbnail,g.min_players,g.max_players,g.min_age,g.rating_average,g.rating_average_weight,g.rating_bayes_average,g.rating_median,g.rating_num_weights,g.rating_stddev,g.users_commented,g.users_owned,g.users_rated,g.users_trading,g.users_wanting,g.users_wishing,', '.join(g.mechanics),', '.join(g.categories),g.description.encode('latin-1','ignore')))

        except mdb.IntegrityError:
            print "Integrity Error, duplicate game id found: ",g.id
    

        con.commit()
        tmpfile.write(str(id)+'\n')
        

    tmpfile.close()
    con.close()
    

        
            
    



if __name__ == "__main__":

    #Creates the game table, IMPORTANT DELETES OLD TABLE
    #####create_games_table() # DELETES OLD GAMES TABLE

    #Retrieves a list containing the game ids in the game_list_file
    #game_list_file = '/Users/davidpuldon/Projects/BGrecommend/bgg1tool/Originalranks1to11027.txt'
    game_list_file = '/Users/davidpuldon/Projects/BGrecommend/bgg1tool/Originalranks1to11027_mod.txt'
    game_id_list = get_game_ids(game_list_file)

    #Call the BGG API to get all the game info for game_ids in game_id_list
    fill_game_table(game_id_list)
    
