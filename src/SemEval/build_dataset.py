import requests
import pandas as pd
url = 'http://tour-pedia.org/api/getReviews?category=accommodation&language=en'

response = requests.get(url)

df = pd.DataFrame(response.json())

del df['language']
del df['rating']
del df['time']
del df['details']
del df['source']