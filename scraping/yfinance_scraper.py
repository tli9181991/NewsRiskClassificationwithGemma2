import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup as bs
import numpy as np
import sys

from selenium import webdriver
from selenium.webdriver.common.by import By

import time

import warnings
warnings.filterwarnings("ignore")

def fetch_yfinance_news_data(ticker, end_datetime = None, max_sroll = 20, slp_time = 10):
    link = f"https://finance.yahoo.com/quote/{ticker}/news/"
    driver = webdriver.Chrome()
    # driver.minimize_window()

    driver.get(link)
    sys.stdout.flush()
    print_str = "\rLoading..."
    sys.stdout.write(print_str)
        
    n_srcoll = 0
    
    END_FATCH = False
    
    if end_datetime != None:
        end_datetime = datetime.strptime(end_datetime.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")
    
    last_height = driver.execute_script('return document.body.scrollHeight')
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    key_sc = driver.find_element(By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[2]/section/div/div/div/div/ul/li')
    key_sc = key_sc.get_attribute('class')
    key_hd = driver.find_element(By.XPATH, '//*[@id="nimbus-app"]/section/section/section/article/section[2]/section/div/div/div/div/ul/li/section/div/a/h3')
    key_hd = key_hd.get_attribute('class').split(' ')[-1]

    print_str = ' '
    while (END_FATCH == False and n_srcoll < max_sroll):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(slp_time)
        
        sys.stdout.flush()
        print_str = '\r' + ''.join([' ' for i in range(len(print_str))])
        sys.stdout.write(print_str)
        
        sys.stdout.flush()
        print_str = f"\rScrolling {n_srcoll}..."
        sys.stdout.write(print_str)
        
        new_height = driver.execute_script('return document.body.scrollHeight')
        
        full_page = bs(driver.page_source, 'html')
        
        last_link = full_page.find_all('li', {'class': key_sc})[-1].find('a').get('href')
        
        content = requests.get(last_link, headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.3"})
        content_bs = bs(content.content, 'html')

        date_time = content_bs.find('time', {'class': 'byline-attr-meta-time'}).text

        weekday, mon_date, year, news_time = date_time.split(',')
        month = mon_date.split(' ')[1]
        day = mon_date.split(' ')[2]
        year = year.replace(' ', '')
        news_date = year + "-" + month + "-" + day

        last_news_datetime = pd.to_datetime(news_date + news_time)
        
        if end_datetime != None:
            delta_s = (last_news_datetime - end_datetime).total_seconds()
            if delta_s < 0:
                END_FATCH = True
                break
                
        if new_height == last_height:
            break
        else:
            # driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            last_height = new_height
            n_srcoll += 1

    print(f"\nTotal Srcoll: {n_srcoll}")

    full_page = bs(driver.page_source, 'html')
    
    sys.stdout.flush()
    print_str = "\rNews Collecting..."
    sys.stdout.write(print_str)

    headlines = [soup.text for soup in full_page.find_all('h3', {'class': key_hd})]
    descriptions = [soup.text for soup in full_page.find_all('p', {'class': key_hd})]
    if (len(headlines) == 0 or len(descriptions) == 0):
        print("ERROR: headlines/descriptions extraction Fail")
    # sources = [soup.text.split(' • ')[0] for soup in full_page.find_all('div', {'class': 'publishing yf-1weyqlp'})]

    news_links = [li_soup.find('a').get('href') for li_soup in full_page.find_all('li', {'class': 'stream-item story-item yf-1drgw5l'})]
    
    news_data = []
    print()
    print_str = ' '
    for headline, description, news_link in zip(headlines, descriptions, news_links):
            
        content = requests.get(news_link, headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.3"})
        content_bs = bs(content.content, 'html')

        sys.stdout.flush()
        print_str = '\r' + ''.join([' ' for i in range(len(print_str))])
        sys.stdout.write(print_str)

        sys.stdout.flush()
        print_str = f'\rprocessing: {news_link}'
        sys.stdout.write(print_str)
        
        if content_bs.find('time', {'class': 'byline-attr-meta-time'}) == None:
            continue

        date_time = content_bs.find('time', {'class': 'byline-attr-meta-time'}).text
        if len(date_time.split(',')) != 4:
           continue
        month = mon_date.split(' ')[1]
        day = mon_date.split(' ')[2]
        year = year.replace(' ', '')
        news_date = year + "-" + month + "-" + day
        news_datetime = pd.to_datetime(news_date + news_time)

        article_soups = content_bs.find('div', {'class': 'atoms-wrapper'})
        if article_soups is not None:
            article = '.'.join([p.text for p in article_soups.find_all('p')])       
        else:
            article = ''

        if end_datetime != None:
            delta_s = (news_datetime - end_datetime).total_seconds()
            if delta_s <= 0:
                break
        
        news_data.append({
                'Datetime': news_datetime,
                'headline': headline,
                'description': description,
                'article': article,
                'link': news_link,
            })
        
    df_news = pd.DataFrame(news_data)
    if len(df_news) > 0:
        print(f"\nNews collected from {news_data[0]['Datetime']} to {news_data[-1]['Datetime']}")    
    else:
        print("No News updated")
    
    driver.close()
    
    return df_news

if __name__ == '__main__':
    # ticker = "APPL"
    for ticker in ['GOOGL', 'META', 'MSFT', 'AMZN', 'TSLA']:
        print(f"Extracting News for {ticker}")
        end_datetime = datetime.now() - timedelta(days=30)
        df_news = fetch_yfinance_news_data(ticker=ticker, end_datetime=end_datetime, 
                                       max_sroll=100, slp_time=10)
        df_news.to_csv(f"./data/yf_news_{ticker}.csv", index=False)