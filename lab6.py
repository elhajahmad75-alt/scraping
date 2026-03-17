import pandas as pd

df = pd.read_csv('ebay_tech_deals.csv')
df.head()

df['effective_price'] = df['price'] + df['shipping']
df['discount_ratio'] = (df['original_price'] - df['price']) / df['original_price']

df['title_len'] = df['title'].apply(len)
df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split()))

df['has_new'] = df['title'].str.contains('new', case=False).astype(int)

df['high_discount'] = df['discount_percentage'] > 20


