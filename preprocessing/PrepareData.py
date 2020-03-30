import json
import requests
import pandas as pd
from tika import parser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def search(uri, size, docName):
    query = json.dumps({
        "from": 0, "size": size,
        "query": {
            "match": {
                "documentName.keyword": docName
            }
        },
      "sort": [
        {
          "sequenceNo": {
            "order": "asc"
          }
        },
        "_score"
      ]
    })
    #print(query)
    headers = {'Content-type': 'application/json'}
    response = requests.get(uri, data=query, headers=headers)
    results = json.loads(response.text)
    return results

def getAllDocumentsWithCount(uri):
    query = json.dumps({
          "size": 0,
          "aggs": {
            "group_by_document": {
              "terms": {
                "field": "documentName.keyword",
                "size": 100000
              }
            }
      }
    })
    headers = {'Content-type': 'application/json'}
    response = requests.get(uri, data=query, headers=headers)
    results = json.loads(response.text)
    return results

def getData(index, _baseUrl):
    _searchUrl = _baseUrl + "/" + index + "/_search"
    docsCountResult = getAllDocumentsWithCount(_searchUrl)
    df = pd.DataFrame(columns=["label", "text"])
    i=0
    for doc in docsCountResult["aggregations"]["group_by_document"]["buckets"]:
        documentName = doc["key"]
        #print("Searching")
        #print(documentName)
        res = search(_searchUrl, doc["doc_count"],documentName)
        sentences = ""
        label = ""
        if len(res["hits"]["hits"]) > 0:
            label = res["hits"]["hits"][0]["_source"]["classification"]
        
        for obj in res["hits"]["hits"]:
            sentence = obj["_source"]["sentence"]
            sentence = sentence.replace('\n', '')
            sentences += sentence + " "
            
        #sentences = "[" + sentences[:-1] + "]"
        df.at[i, 'label'] = label
        df.at[i, 'text'] = sentences
        i+=1
    return df

def readPdfText(filePath):
  raw = parser.from_file(filePath)
  text = raw['content']
  text = preprocessText(text)
  return text

def preprocessText(text):
  #The word_tokenize() function will break our text phrases into #individual words
  tokens = word_tokenize(text)
  #we'll create a new list which contains punctuation we wish to clean
  punctuations = ['(',')',';',':','[',']',',','.']
  #We initialize the stopwords variable which is a list of words like #"The", "I", "and", etc. that don't hold much value as keywords
  stop_words = stopwords.words('english')
  #We create a list comprehension which only returns a list of words #that are NOT IN stop_words and NOT IN punctuations.
  keywords = [word for word in tokens if not word in stop_words and not word in punctuations]

  text = ' '.join(keywords)
  
  return text
    