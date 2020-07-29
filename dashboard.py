import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model


external_stylesheet = ['style.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheet)

## Data
data = pd.read_csv("model.csv", usecols=['popular/poll', 'win', 'inc', 'turnout', 'party'], sep=",")
id1 = pd.read_csv('2020.csv')
id2 = pd.read_csv('map.csv')
polls = px.line(id1, x='Day', y=['Biden', 'Trump'], title='Current Polls:')
spread = px.line(id1,x='Day', y="Spread", title='Current Spread:')
map = go.Figure(data=go.Choropleth(locations=id2['State'],z=id2['Spread'],locationmode = 'USA-states',colorscale = 'Bluered',colorbar_title = "Higher the number, more republican",))
map.update_layout(geo_scope='usa')


## Variables
date = "07/28/20"
accuracy_limit = .7
predict_cand = 'Joe Biden'
predict_poll = 49
predict_inc = 0
predict_turnout = 100
predict_party = 1



## Logic
predict = "win"  # Setting up our predictive value
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])


accuracy = 0
while accuracy_limit > accuracy:
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=.2)
    linear = linear_model.LinearRegression()
    linear.fit(X_train, Y_train)
    accuracy = linear.score(X_test, Y_test)

predictions = linear.predict(X_test)
currentPredictionPoll = linear.coef_[0] * predict_poll
currentPredictionInc = linear.coef_[1] * predict_inc
currentPredictionTurnout = linear.coef_[2] * predict_turnout
currentPredictionParty = linear.coef_[3] * predict_party
currentPrediction = currentPredictionPoll +  currentPredictionInc + currentPredictionTurnout + currentPredictionParty + linear.intercept_
currentPrediction = currentPrediction * 100


## Web Page
app.layout = html.Div(children=[
    html.H1('Election 2020 in Data'),
    html.H2('Current Prediction:'),
    html.H2(f'{predict_cand} has a {currentPrediction}% chance to win the General Election.'),
    html.H6(f"Last updated {date}"),
    dcc.Graph(id='map', figure=map),
    dcc.Graph(id='polls',figure=polls),
    dcc.Graph(id='spread',figure=spread)



    ])

if __name__ == '__main__':
    app.run_server(debug=False)


