from flask import render_template
from app import app
from flask import request
import pymysql as mdb
from recommender_final import *




@app.route('/')
def index():
   user = '' # fake user
   return render_template("index.html",
       title = 'Home',
       user = user)
'''
@app.route('/output')
def call_output():
   user = '' # fake user
   return render_template("output.html",
       title = 'Home',
       user = user)
'''
@app.route('/input')
def cities_input():
  return render_template("input.html")

@app.route('/Contact')
def call_contact():
  return render_template("contact.html")

@app.route('/slide')
def call_slide():
  return render_template("slides.html")


@app.route('/output')
def call_output():
   #pull variables from input field and store it
   usernames = request.args.get('name')
   playtime = request.args.get('playingtime')
   nPlayers = request.args.get('nPlayers')
   hidegames = request.args.get('hidegames')
   #useowner = request.args.get('useowner')
   
   if '&' in usernames:
      g=usernames.replace("&"," ")
      userlist =  g.split()
   else:
      userlist = [usernames]
      
   if hidegames == 0:
      hidegames = False
   else:
      hidegames = True

   nPlayers = int(nPlayers)
   playtime = int(playtime)
   
   if playtime != -1:
      if playtime == 1:
         playtime = 31
      elif playtime == 2:
         playtime = 61
      elif playtime == 3:
         playtime = 91
      elif playtime == 4:
         playtime = 121
      elif playtime == 5:
         playtime = 181


  
   (query_results,query_totalresults)=getPredictions(userlist,-1,nPlayers,playtime,hidegames)

   #with db:
   #  cur = db.cursor()
   #just select the city from the world_innodb that the user inputs
   #  cur.execute("SELECT Name, CountryCode,  Population FROM City WHERE Name='%s';" % city)
   #  query_results = cur.fetchall()
  
   rateonlyresults = []
   for result in query_results:
      rateonlyresults.append(dict(name=result[0].decode('latin-1'), mechanics=result[1], gameid='https://www.boardgamegeek.com/boardgame/'+str(result[2]),thumbnail=result[3],minplayers = result[4],maxplayers = result[5],playtime=result[6],year=result[7]))

   rateownerresults = []
   for result in query_totalresults:
      rateownerresults.append(dict(name=result[0].decode('latin-1'), mechanics=result[1], gameid='https://www.boardgamegeek.com/boardgame/'+str(result[2]),thumbnail=result[3],minplayers = result[4],maxplayers = result[5],playtime=result[6],year=result[7]))
   
   print rateonlyresults[0]['thumbnail']

   #call a function from a_Model package. note we are only pulling one result in the query
   pop_input = 'rawr'
   #the_result = ModelIt(city, pop_input)
   the_result = 'rawr'
   #return render_template("output_new.html", ratedgames = rateonlyresults, rateownergames = rateownerresults)

   return render_template("output_new.html", ratedgames = rateownerresults, rateownergames = rateownerresults)
   '''
   if useowner == "0":
      return render_template("output_new.html", ratedgames = rateownerresults, rateownergames = rateownerresults)
   if useowner == "-1":
      return render_template("output_new.html", ratedgames = rateonlyresults, rateownergames = rateownerresults)
   '''
'''
  #pull 'ID' from input field and store it
  city = request.args.get('ID')

  query_results=getPredictions(city)

  #with db:
  #  cur = db.cursor()
    #just select the city from the world_innodb that the user inputs
  #  cur.execute("SELECT Name, CountryCode,  Population FROM City WHERE Name='%s';" % city)
  #  query_results = cur.fetchall()

  cities = []
  for result in query_results:
    cities.append(dict(name=result[0], country=result[1], population=result[2]))

    #call a function from a_Model package. note we are only pulling one result in the query
  pop_input = 'rawr'
  #the_result = ModelIt(city, pop_input)
  the_result = 'rawr'
  return render_template("output.html", cities = cities, the_result = the_result)
'''
