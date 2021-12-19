#importing libs
from selenium import webdriver
from time import sleep
import pandas as pd

a = []

#setting up webdriver
browser = webdriver.Chrome(executable_path = r"D:\Forsk Files\chromedriver.exe")
browser.maximize_window()

url = "https://www.etsy.com/in-en/c/jewelry-and-accessories?ref=breadcrumb&explicit=1"
browser.get(url)

sleep(2)

#for selecting the page
for pages in range(0, 251):
    item = browser.find_element_by_xpath('//ul[@class="responsive-listing-grid wt-grid wt-grid--block wt-justify-content-flex-start wt-list-unstyled wt-pl-xs-0 tab-reorder-container"]')
    end_prod = len(item.find_elements_by_tag_name('li'))
    print('page no.',pages+1)
    print('No. of products on this page=',end_prod)
#for selecting the product
    try:
        for prod in range(0, (end_prod+1)):
            print('prod no.',prod+1)
            item = browser.find_element_by_xpath('//ul[@class="responsive-listing-grid wt-grid wt-grid--block wt-justify-content-flex-start wt-list-unstyled wt-pl-xs-0 tab-reorder-container"]')
            prod_name = item.find_elements_by_tag_name('li')[prod]
            prod_name.find_element_by_tag_name('a').click()
            sleep(2)
            windows = browser.window_handles
            browser.switch_to.window(windows[1])
#for selecting the review
            try:
                for review_no in range(0,5):
                    review_no = str(review_no)
                    try:
                        review = browser.find_element_by_xpath('//p[@id="review-preview-toggle-' + review_no + '"]')
                        a.append(review.text)
                    except Exception:
                        continue
            except Exception as e:
                print(e)       
            browser.close()
            browser.switch_to.window(windows[0])
    except Exception as e:
        print(e)
    finally:   
        df = pd.DataFrame()   
        df['review'] = a
    
    browser.switch_to.window(windows[0])
    page = browser.find_element_by_xpath('//ul[@class="wt-action-group wt-list-inline"]')
    page = page.find_elements_by_xpath('//li[@class="wt-action-group__item-container"]')[-1]
    page.find_element_by_tag_name('a').click() 
    sleep(2)
    
    
    
scrapped_reviews = df.to_csv('scrapped_reviews')
