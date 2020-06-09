
# Artificial Prospect PipeLinE (APPLE): Assisting the analysis of Hockey Prospects using Recurring Neural Networks
The NHL Draft is the proverbial reset of the NHL calendar. Teams re-evaluate the direction of their organization, make roster decisions, and welcome a new crop of 17-18 year old hockey players. Regardless of the positions team's get to select players, each team's goal is to pick the players most likely to play in the NHL. Most players don't arrive to the NHL untill their early 20s, which leaves teams having to project what a player will be 4-5 years away. This project aims to model a player's development through time given their scoring data and estiamte the possible leagues and performance a player should expect in subsequent seasons. 

## Introduction

A ton of work has been invested on NHL prospect analysis spanning decades of hockey analytics research. Much of the focus concentrates on the following objectives (and are not mutually exclusive in application):
1. Scoring Translations
2. Player Comparables
3. Optimal Draft Value

In my view, these are the foundational concepts and questions that hockey analysts set out to tackle. APPLE is no different — the goal is to expand on the work that's come before, and offer a different approach that addresses shortcomings like:
1. Time dependence (Age)
2. Selection bias
3. Arbitrary response variable (ie. 200 NHL game cutoff)

I wrote about these caveats as limitations of my prospect model last year, with the introduction of the Probability Peers prospects model (PPer). Unlike its predecssors, pGCS / PCS / DEV, the PPer was a regression model since the data span across more leagues than just the CHL, avoiding some selection bias. In the end, I came away still feeling unsastified having not really addressed issues with the response variable, and taking into consideration that each player season is not indepedent violating assumptions of linear regression. It was just a few weeks later my friend Nicole Fitzgerald (Microsoft, soon to be MILA institute), who works in ML research, proposed these issues could be addressed as an LSTM timeseries problem.

That got me thinking.

Most time series problems using Neural Networks, specifically the RNN architecture, leverages deep learning frameworks like LSTM to encode historic information about the timeseries to aid in prediction. Many examples of pytorch/keras tutorials look at Stock Price prediction. Hockey players are assets for their organizations, I thought the analogue was close enough so here we are.  

*DISCLAIMER ON RNN LSTM QUICKLY HERE*




```python
import generate_player_seasons as g

%matplotlib inline
```

`generate_player_seaons` is a python model that we will import to simulate a specifc U23 player

We are going to instantiate the module by invoking the `GeneratePlayer()` method 

## Past Work

After initlizing the player we're reading to simulate a player's development. The `simulate_player_development` method takes the most recent player season (y) and begins by predicting what leagues that player is most likely to play in next year (y+1). Knowing what league a player plays in we can now estimate player performance based on the current season and league in y+1 to get an estiamte of performance in y+1. This process is executed recursively for every predicted league a player is likely to play and stops when they reach 23. 


```python
sim.simulate_player_development()
```

    --- Simulating Seasons --- Vasili Podkolzin --- Age: 19
    --- Simulation Complete --- Vasili Podkolzin --- Age: 20
    --- Simulating Seasons --- Vasili Podkolzin --- Age: 20
    --- Simulation Complete --- Vasili Podkolzin --- Age: 20
    --- Simulating Seasons --- Vasili Podkolzin --- Age: 21
    --- Simulating Seasons --- Vasili Podkolzin --- Age: 21
    --- Simulation Complete --- Vasili Podkolzin --- Age: 21
    --- Simulating Seasons --- Vasili Podkolzin --- Age: 22
    --- Simulating Seasons --- Vasili Podkolzin --- Age: 22
    --- Simulation Complete --- Vasili Podkolzin --- Age: 22


## Methodology

The main philosophical change to APPLE was trying to evaluate player development — treating time as a meaningful dimension in the problem. Not only is player age important in player development, but there are implications on salary cap and asset management as soon a player is drafted. Previous work didn't seem dynamic or global enough to accomplish this. Evaluating a prospect / pick is a careful balance between risk and reward, much like trading Stocks. The goal is to model both risk and reward components separately and to bring it all together at the end to give us a time-dependent value cut off at the age where teams lose entry-level contract rights. APPLE is composed on 3 main models. Using a similar set of player features as the PPer we model the following:
* Predict what League player plays in y+1
* Predict player Scoring conditioned on league in y+1
* Impute remaining features 



```python
import pandas as pd
import sys
sys.path.append('../nhl_development_model/src')
from data_processing import *

df = pd.read_csv('../nhl_development_model/data/player_season_stats.csv')

X, y,_ = prepare_features(df, 'ppg_y_plus_1')
```


```python
X.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>forward</th>
      <th>gp</th>
      <th>gp_y_plus_1</th>
      <th>draft_pick</th>
      <th>is_drafted</th>
      <th>height</th>
      <th>weight</th>
      <th>real_season_age</th>
      <th>gpg</th>
      <th>apg</th>
      <th>ppg</th>
      <th>perc_team_g</th>
      <th>perc_team_a</th>
      <th>perc_team_tp</th>
      <th>AJHL</th>
      <th>Allsvenskan</th>
      <th>BCHL</th>
      <th>CCHL</th>
      <th>Czech</th>
      <th>Czech2</th>
      <th>Jr. A SM-liiga</th>
      <th>KHL</th>
      <th>Liiga</th>
      <th>MHL</th>
      <th>NCAA</th>
      <th>NHL</th>
      <th>NLA</th>
      <th>OHL</th>
      <th>OJHL</th>
      <th>QMJHL</th>
      <th>SHL</th>
      <th>SJHL</th>
      <th>SuperElit</th>
      <th>USHL</th>
      <th>VHL</th>
      <th>WHL</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>round_1.0</th>
      <th>round_2.0</th>
      <th>round_3.0</th>
      <th>round_4.0</th>
      <th>round_5.0</th>
      <th>round_6.0</th>
      <th>round_7.0</th>
      <th>round_8.0</th>
      <th>round_9.0</th>
      <th>next_yr_AJHL</th>
      <th>next_yr_Allsvenskan</th>
      <th>next_yr_BCHL</th>
      <th>next_yr_CCHL</th>
      <th>next_yr_Czech</th>
      <th>next_yr_Czech2</th>
      <th>next_yr_Jr. A SM-liiga</th>
      <th>next_yr_KHL</th>
      <th>next_yr_Liiga</th>
      <th>next_yr_MHL</th>
      <th>next_yr_NCAA</th>
      <th>next_yr_NHL</th>
      <th>next_yr_NLA</th>
      <th>next_yr_OHL</th>
      <th>next_yr_OJHL</th>
      <th>next_yr_QMJHL</th>
      <th>next_yr_SHL</th>
      <th>next_yr_SJHL</th>
      <th>next_yr_SuperElit</th>
      <th>next_yr_USHL</th>
      <th>next_yr_VHL</th>
      <th>next_yr_WHL</th>
    </tr>
    <tr>
      <th>playerid</th>
      <th>player</th>
      <th>season_age</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76532</th>
      <th>Brayden Rose</th>
      <th>17</th>
      <td>0.0</td>
      <td>0.472973</td>
      <td>0.205882</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.447368</td>
      <td>0.92</td>
      <td>0.014815</td>
      <td>0.037037</td>
      <td>0.034815</td>
      <td>0.015086</td>
      <td>0.033803</td>
      <td>0.032573</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>262144</th>
      <th>Tom Hedberg</th>
      <th>17</th>
      <td>0.0</td>
      <td>0.148649</td>
      <td>0.270588</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.52</td>
      <td>0.355263</td>
      <td>0.39</td>
      <td>0.031746</td>
      <td>0.105820</td>
      <td>0.093254</td>
      <td>0.034483</td>
      <td>0.109859</td>
      <td>0.100977</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>399353</th>
      <th>Ty Dellandrea</th>
      <th>17</th>
      <td>1.0</td>
      <td>0.635135</td>
      <td>0.335294</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.60</td>
      <td>0.421053</td>
      <td>0.45</td>
      <td>0.152047</td>
      <td>0.107212</td>
      <td>0.164912</td>
      <td>0.146552</td>
      <td>0.101408</td>
      <td>0.159609</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30930</th>
      <th>Taylor Carnevale</th>
      <th>17</th>
      <td>1.0</td>
      <td>0.635135</td>
      <td>0.341176</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.50</td>
      <td>0.434211</td>
      <td>0.80</td>
      <td>0.070175</td>
      <td>0.068226</td>
      <td>0.089327</td>
      <td>0.068966</td>
      <td>0.064789</td>
      <td>0.084691</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>197821</th>
      <th>Robin Salo</th>
      <th>17</th>
      <td>0.0</td>
      <td>0.472973</td>
      <td>0.100000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.60</td>
      <td>0.407895</td>
      <td>0.22</td>
      <td>0.148148</td>
      <td>0.197531</td>
      <td>0.226296</td>
      <td>0.170259</td>
      <td>0.222535</td>
      <td>0.257329</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X.columns
```




    Index([               'forward',                     'gp',
                      'gp_y_plus_1',             'draft_pick',
                       'is_drafted',                 'height',
                           'weight',        'real_season_age',
                              'gpg',                    'apg',
                              'ppg',            'perc_team_g',
                      'perc_team_a',           'perc_team_tp',
                             'AJHL',            'Allsvenskan',
                             'BCHL',                   'CCHL',
                            'Czech',                 'Czech2',
                   'Jr. A SM-liiga',                    'KHL',
                            'Liiga',                    'MHL',
                             'NCAA',                    'NHL',
                              'NLA',                    'OHL',
                             'OJHL',                  'QMJHL',
                              'SHL',                   'SJHL',
                        'SuperElit',                   'USHL',
                              'VHL',                    'WHL',
                                 18,                       19,
                                 20,                       21,
                                 22,                       23,
                        'round_1.0',              'round_2.0',
                        'round_3.0',              'round_4.0',
                        'round_5.0',              'round_6.0',
                        'round_7.0',              'round_8.0',
                        'round_9.0',           'next_yr_AJHL',
              'next_yr_Allsvenskan',           'next_yr_BCHL',
                     'next_yr_CCHL',          'next_yr_Czech',
                   'next_yr_Czech2', 'next_yr_Jr. A SM-liiga',
                      'next_yr_KHL',          'next_yr_Liiga',
                      'next_yr_MHL',           'next_yr_NCAA',
                      'next_yr_NHL',            'next_yr_NLA',
                      'next_yr_OHL',           'next_yr_OJHL',
                    'next_yr_QMJHL',            'next_yr_SHL',
                     'next_yr_SJHL',      'next_yr_SuperElit',
                     'next_yr_USHL',            'next_yr_VHL',
                      'next_yr_WHL'],
          dtype='object')



## Model Architecture

With a NetworkX Directed Graph created from our original player season, we can visualize this as a tree diagram and see what a player's likely player development path will take, and perhaps what the optimal path is to maximize NHL production


```python

```

## Data Processing 


```python
from full_data_load_ep import *
# load database credentials and create connection
user, password, server, database, port = load_db_credentials()
engine = create_engine(f'postgresql://{user}:{password}@{server}:{port}/{database}')
```


```python
print('--- Reading Data From Database ---')
# read data in from database
query = open('../prospect_database/player_stats.sql', 'r').read()

skaters = pd.read_sql(query, engine)
team_stats = pd.read_sql(''' select * from team_stats  ''', engine)
info = pd.read_sql(''' select * from player_info ''', engine)

print('--- Engineering features ---')

# merge player information with player seasons
skaters = skaters.merge(info, on = ['playerid'])

# get season age per player season
skaters['start_year'], skaters['end_year']  = zip(*skaters['year'].apply(lambda x : x.split('-')))
skaters = get_season_age(skaters)

# return player season with one league played per 
# year. Choose league where player played most games
df = collapse_player_league_seasons(skaters)

# split data into X, y
X, y,_ = prepare_features(df, 'ppg_y_plus_1')

players = X.index.droplevel(-1).unique() # get number indices
n_players = players.shape[0] # get number of players
train_idx, test_idx = train_test_split(players, test_size=0.3, random_state=42)

print('--- Padding Player Data ---')

X = pad_data(X.reset_index(), players)
y = pad_data(y.reset_index(), players)

X.set_index(['playerid', 'player', 'season_age'], inplace=True)
y.set_index(['playerid', 'player', 'season_age'], inplace=True)

print('--- Generating Player Data ---')
train_seq, train_target = generate_players(X, y, train_idx)
test_seq, test_target = generate_players(X, y, test_idx)

```

    --- Reading Data From Database ---
    --- Engineering features ---
    --- Padding Player Data ---
    --- Generating Player Data ---


## Results

With the models trained, we can now start our process for predicting player seasons recursively calculating leagues and performance reaching the base case when a player reaches the 23 year old season. This ensures that each simulated season created acts as its own node. This creates these independent timelines across nodes at each age, and that node is coniditioned on just one node. With that we can calculate the NHL likelihood at age 23 since all the logits sum to 1. This gives us a level of risk, for the reward side of the equation we're calculating the production a player would expect at each NHL season. We get our Expected Value by summing all the products of NHL expected points at age 23 by the Conditional Probability of that node.  

Let's look at an example, Alex Newhook (one of my favourite prospects from last year's draft) is an 19 year old prospect who just finished they Draft + 1 season in the NCAA. We pass this past season into APPLE to simualte his 20 year old season, which produces three possible outcomes {NHL, AHL, NCAA} based on his scoring and other player features. We can then estimate his scoring and the process repeats itself untill he reaches his hypothetical 23 year old season.

APPLE thinks Alex Newhook has an _87%_ chance to make the NHL by 23. At that strong a likelihood to play in the NHL, his expected NHL Value over in the 5 seasons since being drafted is _~135.7 points_.


```python
from generate_player_seasons import GeneratePlayer

player = GeneratePlayer()

player.initialize_player('320307')
player.simulate_player_development()
player.generate_network_graph()
player.plot_network_graph()
```

    [19:45:32] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    --- Simulating Seasons --- Alex Newhook --- Age: 19
    --- Simulation Complete --- Alex Newhook --- Age: 20
    --- Simulating Seasons --- Alex Newhook --- Age: 20
    --- Simulation Complete --- Alex Newhook --- Age: 20
    --- Simulating Seasons --- Alex Newhook --- Age: 21
    --- Simulating Seasons --- Alex Newhook --- Age: 21
    --- Simulation Complete --- Alex Newhook --- Age: 21
    --- Simulating Seasons --- Alex Newhook --- Age: 22
    --- Simulating Seasons --- Alex Newhook --- Age: 22
    --- Simulation Complete --- Alex Newhook --- Age: 22



![png](Readme_files/Readme_21_1.png)



```python
player.player_value
```




    {'playerid': '320307',
     'player_name': 'Alex Newhook',
     'nhl_likelihood': 0.87,
     'most_likelihood_nhl_node': 21,
     'nhl_expected_value': 135.7}



## EDA


```python

```

## Limitations


```python

```

## Closing Thoughts


```python

```
