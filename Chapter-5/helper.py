import pickle
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

stop = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours',
        'yourself','yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its',
        'itself','they','them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",
        'these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does',
        'did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for',
        'with','about','against','between','into','through','during','before','after','above','below','to','from',
        'up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where',
        'why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own',
        'same','so','than','too','very','s','t','can','will','just','don',"don't",'should',"should've",'now','d','ll',
        'm','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",
        'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',
        "shan't", 'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]

label_mapper = {"Celebrity":0, "Chemistry":1, "Physics":2,
               "Mathematics":3, "Statistics":4, "Economics":5}

def get_wiki_data(url):
  page = requests.get(url)
  html = page.text
  soup = BeautifulSoup(html)
  content = soup.find_all("p")
  data = []
  for paragraph in content:
    data.append(paragraph.text)
  data = " ".join(data)
  return data

def clean_text(text):
  text = text.lower()
  tokens = text.split()
  new = [word for word in tokens if word not in stop]
  text = " ".join(new)
  text = " ".join(re.split(r"[^A-Za-z]", text))
  text = re.sub(" +", " ", text)
  return(text)
def from_url_to_predict_data(url):
  text = get_wiki_data(url)
  cleaned_text = clean_text(text)
  data_for_prediction = pd.Series(cleaned_text)
  return data_for_prediction

def get_prediction(url,model):
  req_data = from_url_to_predict_data(url)
  pred = model.predict(req_data)
  return [name for name,code in label_mapper.items() if code == pred][0]

with open("wiki_model.bin", "rb") as file:
    pipeline = pickle.load(file)


# test_url = r"https://en.wikipedia.org/wiki/Jordan_Peterson"
# test_url2 = r"https://en.wikipedia.org/wiki/Cosmos"
# print(get_prediction(test_url, pipeline))
# print(get_prediction(test_url2, pipeline))