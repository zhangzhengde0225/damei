import damei as dm

dm.ColorControl().r()

from bs4 import BeautifulSoup
import bs4
import urllib3
import json

url = "https://jbox.sjtu.edu.cn/l/snpjWy"
url = "https://jbox.sjtu.edu.cn/l/R09DWP"
http = urllib3.PoolManager()

r = http.request('GET', url=url, headers='')
data = r.data.decode()

print(data, len(data))

a = '<iframe src='
print(a in data)
