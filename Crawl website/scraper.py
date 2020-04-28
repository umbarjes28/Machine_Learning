import requests
from bs4 import BeautifulSoup
import mysql.connector

# request link
page = requests.get('https://www.imdb.com/list/ls069887650/')
soup = BeautifulSoup(page.text, 'html.parser')
#find given class in soup object
artist_name_list = soup.find(class_='article listo')
name = artist_name_list.find_all('img')
details = artist_name_list.find_all(class_='lister-item-content')
#declared arrays for name of actors, image link & details
names = []
images = []
data = []

#storing data into arrays
for artist_name in name:
    names.append(artist_name['alt'])
    images.append(artist_name['src'])

#storing data into array
for d in details:
    val = d.find_all('p')
    data.append(val[1].contents[0])

#connecting to database

db_connection = mysql.connector.connect(
  host="localhost",
  user="root",
  password="####",
  auth_plugin='mysql_native_password',
  database='actors',
)

mycursor = db_connection.cursor()
#inserting all records into actor table
for x in range(len(names)):
    if names[x] != "" and images[x] != "" and data[x] != "":
        val = (str(names[x]), str(images[x]), str(data[x]))
        sql= """INSERT INTO actor(name, image_link, details) VALUES(%s,%s,%s)"""
        mycursor.execute(sql,val)
        db_connection.commit()
        
print(mycursor.rowcount, "record inserted.")