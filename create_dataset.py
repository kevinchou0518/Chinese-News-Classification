
import requests
import json 
from tqdm import tqdm
from bs4 import BeautifulSoup  
from datetime import datetime


# Function to check if a date is the same as a specific date
def is_on_specific_date(date_str, specific_date_str):
    # Parse the date from the string
    date = datetime.strptime(date_str, "%Y/%m/%d %H:%M")
    # Parse the specific date to compare
    specific_date = datetime.strptime(specific_date_str, "%Y/%m/%d")
    # Check if the date from the string is the same as the specific date
    return date.date() == specific_date.date()


"""
政治、體育、財經、遊戲、影劇
  1    10   17    24    9
"""

# List of news types
typelist = [1,10,17,24,9]

# Lists to store the data
titles = []
classtypes = []

# Accessing all news headlines and urls from 2023/1/1 to 2023/12/31
print("Accessing all news headlines and urls from 2023/1/1 to 2023/12/31...")
for year in tqdm(range(2023,2024)):
    for month in tqdm(range(1,13)):
        for day in tqdm(range(1,32)):
            for type in typelist:
                u = 'https://www.ettoday.net/news/news-list-'+str(year) \
                    +'-'+str(month)+'-'+str(day)+'-'+str(type)+'.htm'
                try:
                    res = requests.get(u)
                    soup = BeautifulSoup(res.content, "lxml")
                    soup = soup.find("div", class_="part_list_2")
                    for a in soup.find_all("h3"):
                        #print(a.span.string)
                        if(is_on_specific_date(a.span.string, str(year)+'/'
                        +str(month)+'/'+str(day)) and a.a.string != None):   
                            titles.append(a.a.string)
                            classtypes.append(a.em.string)
                except:
                    continue

# Dump the data into a json file
js_dataset = []
with open('dataset.json', 'w', encoding='UTF-8') as dataset:
    print("arrange the data and dump into json file")
    for index, (title, classtype) in tqdm(enumerate(zip(titles, classtypes))):
        if index >= 50000: # Limit the number of data samples
            break
        js_dataset.append({"id" : index, "title" : title, "class" : classtype})
    print("dump to train.json")
    for data in js_dataset:
        dataset.write(json.dumps(data , ensure_ascii=False) + "\n")
print("Finish")