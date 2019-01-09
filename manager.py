import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

city_to_country = {
        'zagreb': 'croatia',
        'berlin': 'germany',
        'poznan': 'poland',
        'warszaw': 'poland',
        'gdynia': 'poland',
        'gdansk': 'poland',
        'sopot': 'poland',
        'krakow': 'poland',
        'wroclaw': 'poland',
        'malmo': 'sweden',
        'gothenburg': 'sweden',
        'vasteras': 'sweden',
        'stockholm': 'sweden',
        'copenhagen': 'denmark',
        'prague': 'czechia',
        'bergamo': 'italy',
        'milano': 'italy',
        }

country_to_currency = {
        'croatia': 'HRK',
        'poland': 'PLN',
        'italy': 'EUR',
        'germany': 'EUR',
        'sweden': 'SEK',
        'denmark': 'DKK',
        'czechia': 'CZK',
        }

rates = {
        ('PLN', 'HRK'): 1.73,
        ('EUR', 'HRK'): 7.43,
        ('CZK', 'HRK'): 0.29,
        ('HRK', 'EUR'): 0.13,
    }

def get_rate(fromc, toc, date):
    if fromc==toc:
        return 1
    return rates[fromc, toc]

def transform_row(r):
    if len(r.date) == 6:
        r.date += '2018.'
    d = r.date[:-1].split('.')
    r.date = date(*map(int, d[::-1]))
    r.country = city_to_country[r.city]
    r.currency = country_to_currency[r.country]
    if np.isnan(r.hrk):
        r.hrk = r.lcy * get_rate(r.currency, 'HRK', r.date)
    r.eur = r.hrk * get_rate('HRK', 'EUR', r.date)
    return r

df = pd.read_csv('./expenses.csv')
df = df.apply(transform_row, axis=1)

'''
category_sum = []
for category, rows in df.groupby(['category'])['eur']:
    category_sum.append((sum(rows.values), category))
sums, labels = zip(*sorted(category_sum, reverse=True)[:11])
explode = [0.1]*len(sums)

fig1, ax1 = plt.subplots()
ax1.pie(sums, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')
plt.title('percentage of money spend on each category')
plt.show()
'''

'''
preferred_transport = []
for desc, rows in df.groupby(['description']):
    if all(i in ['travel', 'transport'] for i in rows['category']):
        preferred_transport.append((sum(rows['eur'].values), desc))

sums, labels = zip(*sorted(preferred_transport, reverse=True))
explode = [0.1]*len(sums)

fig1, ax1 = plt.subplots()
ax1.pie(sums, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')
plt.title('preferred transport')
plt.show()
'''

'''
all_categories = tuple(set(df['category']) - set('travel'))
cities_daily = []
for city, rows in df.groupby(['city']):
    days = set(rows['date'].values)
    days = (max(days) - min(days)).days + 1
    descs = {desc: sum(rs['eur'].values)/days for desc, rs in rows[rows['category'] != 'travel'].groupby(['category'])}
    cities_daily.append((city, tuple(descs[i] if i in descs else 0 for i in all_categories)))

cities, sums = zip(*sorted(cities_daily, reverse=True, key=lambda t: sum(t[1])))
sums = list(zip(*sums))

ind = np.arange(len(cities))
width = 0.35
colors = ['maroon','c','orange','k','b','darkmagenta','g','m','yellow','r','peru','navy','cyan','plum','grey','teal','lime']
bars = [plt.bar(ind, sums[0], width, color=colors[0])]
for i in range(1, len(all_categories)):
    bars.append(plt.bar(ind, sums[i], width, bottom=list(map(sum, zip(*sums[:i]))), color=colors[i]))

plt.title('amount of money spent daily per city')
plt.xticks(ind, cities)
plt.yticks(np.arange(0, 26, 1))
plt.legend(list(zip(*bars))[0], all_categories)
plt.show()
'''

daily_expenses = []
all_dates = list(pd.date_range(min(df['date']), max(df['date']), freq='D'))
cities = []
for d in list(all_dates):
    value = sum(df[df['date'] == d.date()]['eur'])
    if value:
        cities.append(df[df['date'] == d.date()]['city'].values[-1])
        daily_expenses.append((d.date(), value))
    else:
        all_dates.remove(d)
dates, sums = zip(*daily_expenses)

'''
ind = np.arange(len(all_dates))
plt.bar(ind, sums, color='red', width=0.35)
plt.xticks(ind, list(range(len(all_dates))))
plt.title('daily amount of money spend')
plt.xlabel('day number')
plt.ylabel('amount of money in eur')
plt.show()
'''

# encoding strings
x = np.array([*zip(range(len(dates)), cities)])
y = sums
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
preprocess = make_column_transformer((OneHotEncoder(), [-1])).fit_transform(x)
x = np.array([*zip(preprocess, x[:, 0])])

# avoiding the dummy variable trap
x = x[:, 1:]

# splitting into test set and training set
from sklearn.model_selection import train_test_split as tts
xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.2)

# fitting the regressor to our training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# applying the regressor to our test set
ypred = regressor.predict(xtest)

# backward elimination
import statsmodels.formula.api as sm
xopt = np.hstack([np.ones((x.shape[0], 1)), x])
for i in range(xopt.shape[1]):
    pvalues = sm.OLS(y, xopt.astype(np.float64)).fit().pvalues
    mi = np.argmax(pvalues)
    mp = pvalues[mi]
    if mp > 0.05:
        xopt = np.delete(xopt, [mi], 1)
    else:
        break

xtrain, xtest, ytrain, ytest = tts(xopt, y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

ypredopt = regressor.predict(xtest)

plt.plot(ytest, color = 'green')
plt.plot(ypred, color = 'navy')
plt.plot(ypredopt, color = 'red')
plt.ylabel('predicted value in eur')
plt.xlabel('days in the test set')
plt.show()
