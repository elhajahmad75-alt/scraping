from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import pandas as pd
import time
from datetime import datetime
import os

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

url = "https://www.ebay.com/globaldeals/tech"

driver = webdriver.Chrome()
driver.get(url)

time.sleep(5)

last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(3)

    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height



products = driver.find_elements(By.CLASS_NAME, "dne-itemtile")
    

data = []

for product in products:

    timestamp = datetime.now()

    try:
        title = product.find_element(By.CLASS_NAME,"dne-itemtile-title").text
    except:
        title = "N/A"

    try:
        price = product.find_element(By.CLASS_NAME,"dne-itemtile-price").text
    except:
        price = "N/A"

    try:
        original_price = product.find_element(By.CLASS_NAME,"dne-itemtile-original-price").text
    except:
        original_price = "N/A"

    try:
        shipping = product.find_element(By.CLASS_NAME,"dne-itemtile-delivery").text
    except:
        shipping = "N/A"

    try:
        item_url = product.find_element(By.TAG_NAME,"a").get_attribute("href")
    except:
        item_url = "N/A"

    data.append([timestamp,title,price,original_price,shipping,item_url])



    columns = ["timestamp","title","price","original_price","shipping","item_url"]

df = pd.DataFrame(data,columns=columns)

file_name = "ebay_tech_deals.csv"

if os.path.exists(file_name):
    df.to_csv(file_name,mode="a",header=False,index=False)
else:
    df.to_csv(file_name,index=False)

driver.quit()





