# scrapper to get police budgets for police budgets in america
from lxml import html
import requests
import pandas as pd
import threading

OUTPUT_CSV = 'data/city-budgets.csv'
CITY_DATA_URL = 'http://www.city-data.com'

columns = (
    'State', 
    'City', 
    'Full-time employees', 
    'Monthly full-time payroll', 
    'Average yearly full-time wage', 
    'Part-time employees', 
    'Monthly part-time payroll')
df = pd.DataFrame(columns=columns)

def get_tree(url):
    page = requests.get(url)
    return html.fromstring(page.content)

tree = get_tree(CITY_DATA_URL)
state_urls = tree.xpath('//*[@id="tabs_by_category"]//div[@id="home1"]//li//a/@href')

def get_city_urls(tree):
    city_links = tree.xpath('//*[@id="cityTAB"]//td//a[contains(@href,"html")]/@href')
    abs_links = list(map(lambda link: "/".join((CITY_DATA_URL, "city", link)), city_links))
    return abs_links

def get_budgets(tree):
    table_headings = tree.xpath('//*[@id="government-employment"]//thead//th/text()')
    values = tree.xpath('//*[@id="government-employment"]//tfoot//td/text()')
    row = {}
    for index, heading in enumerate(table_headings):
        row[heading] = values[index]
    return row

def get_state_name(tree):
    return tree.xpath('//*[@class="banner"]//span/text()')[0]

def get_city_name(tree):
    name = tree.xpath('//*[@id="content"]//h1[@class="city"]//span/text()')[0]
    return name[:name.find(",")]

def generate_rows(state_url):
    global df
    print("Generate rows")
    state_tree = get_tree(state_url)
    state_name = get_state_name(state_tree)
    city_urls = get_city_urls(state_tree)
    for url in city_urls[:1]:
        print('[*] City task: fetching %s' % url)
        city_tree = get_tree(url)
        city_name = get_city_name(city_tree)
        row = get_budgets(city_tree)
        cols = [state_name, city_name] + list(row.values())[1:]
        if len(cols) == 7:
            df.loc[len(df)] = cols
    pool.release()
    print("Active threads: %s" % threading.active_count())


pool = threading.BoundedSemaphore(2)
        
for url in state_urls[:1]:
    print('[*] State task: fetching %s' % url)
    pool.acquire(blocking=True)
    thread = threading.Thread(target=generate_rows, args=(url, ))
    thread.start()
    # DEV
    # rows = generate_rows(url)
    # callback(rows)


print("Finished")
df.to_csv(OUTPUT_CSV)
