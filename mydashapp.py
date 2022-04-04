import pandas as pd
# import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# from jupyter_dash import JupyterDash
from dash import dcc, html, Dash, State
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input

# IMPORTING AND MERGING THE DATA
# dfBasics = pd.read_csv('title.basics.tsv', sep='\t', header=0)
# dfRegion = pd.read_csv('Downloads/title.akas.tsv', sep='\t', header=0)
# dfRatings = pd.read_csv('title.ratings.tsv', sep='\t', header=0)

# dfTotal = pd.merge(dfBasics, dfRatings, on='tconst')

# LOADING MOVIES
# dfMovies = dfTotal.loc[dfTotal['titleType'] == 'movie']
# dfMovies.rename({'startYear': 'year'}, axis=1, inplace=True)
# dfMovies = dfMovies[dfMovies.runtimeMinutes != '\\N']
# dfMovies = dfMovies[dfMovies.genres != '\\N']
# dfMovies = dfMovies[dfMovies.year != '\\N']
# dfMovies = dfMovies[dfMovies['numVotes'].astype(int) >= 25000]
# Fixing movies so that the genres are split into different rows
# dfMovies.drop(['originalTitle'], axis=1, inplace=True)
# dfMovies.drop(['endYear'], axis=1, inplace=True)
# dfMovies.set_index(['tconst'], inplace=True)

# dfMovies['genres'] = dfMovies['genres'].str.split(',')
# dfMovies = dfMovies.explode('genres')
# dfMovies = dfMovies[dfMovies.genres != 'Game-Show']

# LOADING TV SHOWS
# dfShows = dfTotal.loc[dfTotal['titleType'].isin(['tvSeries', 'tvMiniSeries'])]
# dfShows = dfShows[dfShows.runtimeMinutes != '\\N']
# dfShows = dfShows[dfShows.genres != '\\N']
# dfShows = dfShows[dfShows.startYear != '\\N']
# dfShows = dfShows[dfShows['numVotes'].astype(int) >= 10000]
# dfShows.endYear.replace('\\N', 2022, inplace=True)  # REMOVE IF I WANNA SCRAP THE CHANGE OF \N to 2022
# Fixing the TV Shows database so that genres are split into different rows
# dfShows.drop(['originalTitle'], axis=1, inplace=True)
# dfShows.set_index(['tconst'], inplace=True)

# dfShows['genres'] = dfShows['genres'].str.split(',')
# dfShows = dfShows.explode('genres')

# dfShows.to_csv('dfShows.csv', index = False)
# dfMovies.to_csv('dfMovies.csv', index = False)


# IMPORTING DATA AFTER MANIPULATION
dfShows = pd.read_csv('dfShows.csv')
dfMovies = pd.read_csv('dfMovies.csv')

dfShows['year'] = dfShows['startYear'].astype(str) + '-' + dfShows['endYear'].astype(str)

# Main Page found below
#external_stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH],
           meta_tags=[{'name':'viewport', 'content':'width=device-width, initial-scale=1.0,'
                                                   'maximum-scale=1.2, minimum-scale=0.5'}]
           )
server = app.server


# Function for TV Show year Range
def filterTVyears(df, low, high, genre=None):
    if genre is None:
        mask = ((df['startYear'].astype(int) >= low) & (df['startYear'].astype(int) <= high)) | \
               ((df['endYear'].astype(int) <= high) & (df['endYear'].astype(int) >= low)) | \
               ((df['startYear'].astype(int) >= low) & (df['endYear'].astype(int) <= high)) | \
               ((df['startYear'].astype(int) <= low) & (df['endYear'].astype(int) >= high))
    else:
        mask = (((df['startYear'].astype(int) >= low) & (df['startYear'].astype(int) <= high)) | \
                ((df['endYear'].astype(int) <= high) & (df['endYear'].astype(int) >= low)) | \
                ((df['startYear'].astype(int) >= low) & (df['endYear'].astype(int) <= high)) | \
                ((df['startYear'].astype(int) <= low) & (df['endYear'].astype(int) >= high))) & \
               (df['genres'].astype(str) == genre)

    return mask


# BUBBLE GRAPH UPDATING
@app.callback(
    Output(component_id='output_container', component_property='children'),
    Output(component_id='bubble', component_property='figure'),
    Input(component_id='TVorMovie', component_property='value'),
    Input(component_id='year_slider', component_property='value')
)
def update_graph(option_slct, year_slct):
    low, high = year_slct

    if option_slct == 1:
        if low == high:
            container = 'You are currently looking at {} in {}'.format('Movies', high)

        else:
            container = 'You are currently looking at {} between the years {} and {}'.format('Movies', low, high)
        bubble_count = dfMovies.copy()
        mask_count = (bubble_count['year'].astype(int) >= low) & (bubble_count['year'].astype(int) <= high)
    else:
        if low == high:
            container = 'You are currently looking at {} in {}'.format('TV Shows', high)

        else:
            container = 'You are currently looking at {} between the years {} and {}'.format('TV Shows', low, high)
        bubble_count = dfShows.copy()
        mask_count = filterTVyears(bubble_count, low, high)

    fig = px.scatter(
        bubble_count[mask_count].groupby('genres').mean(), x='numVotes', y='averageRating',
        size=bubble_count[mask_count].groupby('genres')['numVotes'].count(),
        color=bubble_count[mask_count].groupby('genres').genres.first(),
        hover_name=bubble_count[mask_count].groupby('genres').genres.first(),
        custom_data=[bubble_count[mask_count].groupby('genres').genres.first()],
        labels=dict(numVotes="Average Number of Votes", averageRating="Average Rating out of 10",
                    size="Number of releases", color='Genre'),
        log_x=False, size_max=65,
        title='Number of votes vs Average Rating out of 10 Filtered by Genre and/or Year(s)',
        template='plotly_dark'
    )

    fig.update_layout(clickmode='event+select')

    return container, fig


# VIOLIN GRAPH UPDATING
@app.callback(
    Output(component_id='violin', component_property='figure'),
    Input(component_id='TVorMovie', component_property='value'),
    Input(component_id='year_slider', component_property='value'),
    Input(component_id='bubble', component_property='selectedData'),
    Input(component_id='RateOrPop', component_property='value')
)
def update_violin(option_slct, year_slct, clicked_genre, rateorpop):
    low, high = year_slct

    if clicked_genre is None:

        if option_slct == 1:

            violin_count = dfMovies.copy()
            mask_count = (violin_count['year'].astype(int) >= low) & (violin_count['year'].astype(int) <= high)

            violin_av = violin_count[mask_count].groupby('genres').mean()
            violin_av.reset_index(inplace=True)
        else:

            violin_count = dfShows.copy()
            mask_count = filterTVyears(violin_count, low, high)

            violin_av = violin_count[mask_count].groupby('genres').mean()
            violin_av.reset_index(inplace=True)

        if rateorpop == 1:
            fig = px.violin(
                violin_av, y='averageRating', box=True, points='all',
                labels=dict(numVotes="Average Number of Votes", averageRating="Average Rating out of 10",
                            size="Total Count", genres='Genre'), title='KDE of genres in the given years',
                hover_name=violin_av.genres,
                hover_data=['averageRating', 'numVotes'],
                template='plotly_dark',
                color_discrete_sequence=['#e0b416']
            )
        else:
            fig = px.violin(
                violin_av, y='numVotes', box=True, points='all',
                labels=dict(numVotes="Average Number of Votes", averageRating="Average Rating out of 10",
                            size="Total Count", genres='Genre'), title='KDE of genres in the given years',
                hover_name=violin_av.genres,
                hover_data=['averageRating', 'numVotes'],
                template='plotly_dark',
                color_discrete_sequence=['#e0b416']
            )
    else:
        genre = clicked_genre['points'][0]['customdata'][0]  # How do we get multi-select?

        if option_slct == 1:
            violin_count = dfMovies.copy()
            mask_count = (violin_count['year'].astype(int) >= low) & (violin_count['year'].astype(int) <= high) & \
                         (violin_count['genres'].astype(str) == genre)

        else:
            violin_count = dfShows.copy()
            mask_count = filterTVyears(violin_count, low, high, genre)

        if rateorpop == 1:
            fig = px.violin(
                violin_count[mask_count], y='averageRating', box=True, points='all',
                labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                            size="Total Count", runtimeMinutes='Runtime in Minutes', year='Year(s)'),
                title='KDE of the {} genre'.format(genre),
                hover_name=violin_count[mask_count].primaryTitle,
                hover_data=['averageRating', 'numVotes', 'runtimeMinutes', 'year'],
                template='plotly_dark',
                color_discrete_sequence=['#e0b416']
            )
        else:
            fig = px.violin(
                violin_count[mask_count], y='numVotes', box=True, points='all',
                labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                            size="Total Count", runtimeMinutes='Runtime in Minutes',
                            year='Year(s)'),
                title='KDE of the {} genre'.format(genre),
                hover_name=violin_count[mask_count].primaryTitle,
                hover_data=['averageRating', 'numVotes', 'runtimeMinutes', 'year'],
                template='plotly_dark',
                color_discrete_sequence=['#e0b416']
            )

    return fig


# RUNTIME GRAPH UPDATING   #TOGGLE BETWEEN MOST VOTES AND TOP RANKS
@app.callback(
    Output(component_id='runTime', component_property='figure'),
    Input(component_id='TVorMovie', component_property='value'),
    Input(component_id='year_slider', component_property='value'),
    Input(component_id='bubble', component_property='selectedData'),
    Input(component_id='RateOrPop', component_property='value')
)
def update_RUN(option_slct, year_slct, clicked_genre, rateorpop):
    low, high = year_slct

    if clicked_genre is None:

        if option_slct == 1:
            run_count = dfMovies.copy()
            mask_count = (run_count['year'].astype(int) >= low) & (run_count['year'].astype(int) <= high)

        else:
            run_count = dfShows.copy()
            mask_count = filterTVyears(run_count, low, high)
    else:

        genre = clicked_genre['points'][0]['customdata'][0]  # How do we get multi-select?

        if option_slct == 1:

            run_count = dfMovies.copy()
            mask_count = (run_count['year'].astype(int) >= low) & (run_count['year'].astype(int) <= high) & (
                    run_count['genres'].astype(str) == genre)

        else:

            run_count = dfShows.copy()
            mask_count = filterTVyears(run_count, low, high, genre)

    run_count.runtimeMinutes = run_count.runtimeMinutes.astype(int)
    run_count2 = run_count[mask_count].sort_values(by='runtimeMinutes', ascending=True)
    run_count = run_count2.groupby(['primaryTitle', 'runtimeMinutes'], as_index=False).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(template='plotly_dark')


    # create two independent figures with px.line each containing data from multiple columns
    fig0 = px.histogram(run_count2, x='runtimeMinutes', template='plotly_dark', color_discrete_sequence=['#e0b416'])
    fig0.update_traces(yaxis='y1')

    if rateorpop == 1:
        fig1 = px.line(run_count.groupby('runtimeMinutes', as_index=False).mean(),
                       x='runtimeMinutes', y='averageRating', render_mode="webgl", template='plotly_dark')
        fig1.update_traces(line=dict(color="#e25681"), yaxis='y2')

        fig.add_traces(fig0.data + fig1.data)
        fig.layout.title = 'Runtime Trends'
        fig.layout.xaxis.title = "Runtime in Minutes"
        fig.layout.yaxis.title = "Count"
        fig.layout.yaxis2.title = "Average Rating out of 10"
    else:
        fig1 = px.line(run_count.groupby('runtimeMinutes', as_index=False).mean(),
                       x='runtimeMinutes', y='numVotes', render_mode="webgl", template='plotly_dark')
        fig1.update_traces(line=dict(color="#e25681"), yaxis='y2')

        fig.add_traces(fig0.data + fig1.data)
        fig.layout.title = 'Runtime Trends'
        fig.layout.xaxis.title = "Runtime in Minutes"
        fig.layout.yaxis.title = "Count"
        fig.layout.yaxis2.title = "Number of votes"

    return fig


# RANKINGS GRAPH UPDATING   #TOGGLE BETWEEN MOST VOTES AND TOP RANKS

@app.callback(
    Output(component_id='toprank', component_property='figure'),
    Input(component_id='TVorMovie', component_property='value'),
    Input(component_id='year_slider', component_property='value'),
    Input(component_id='bubble', component_property='selectedData'),
    Input(component_id='RateOrPop', component_property='value')
)
def update_ranks(option_slct, year_slct, clicked_genre, rateorpop):
    low, high = year_slct

    if clicked_genre is None:

        if option_slct == 1:
            minVotes = 25000
            rank_count = dfMovies.copy()
            mask_count = (rank_count['year'].astype(int) >= low) & (rank_count['year'].astype(int) <= high)

        else:
            minVotes = 10000
            rank_count = dfShows.copy()
            mask_count = filterTVyears(rank_count, low, high)

        topRank = rank_count[mask_count]
        topRank = topRank.groupby(['primaryTitle', 'numVotes'], as_index=False).agg(
            {'averageRating': 'first', 'genres': lambda x: ','.join(x.astype(str))})

        if rateorpop == 1:
            topRank = topRank.nlargest(10, 'averageRating')
            topRank.sort_values(by='averageRating', inplace=True)

            fig = px.bar(
                topRank, y='primaryTitle', x='averageRating', color='numVotes', orientation='h',
                labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                            primaryTitle="Title", genres='Genre'),
                title='Top 10, minimum {} votes'.format(minVotes),
                hover_name=topRank.primaryTitle,
                hover_data=['averageRating', 'numVotes', 'genres'],
                color_continuous_scale='pinkyl',
                template='plotly_dark'
            )
        else:
            topPop = topRank.nlargest(10, 'numVotes')
            topPop.sort_values(by='numVotes', inplace=True)

            fig = px.bar(
                topPop, y='primaryTitle', x='numVotes', color='averageRating', orientation='h',
                labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                            primaryTitle="Title", genres='Genre'),
                title='Top 10, minimum {} votes'.format(minVotes),
                hover_name=topPop.primaryTitle,
                hover_data=['averageRating', 'numVotes', 'genres'],
                color_continuous_scale='pinkyl',
                template='plotly_dark'
            )
    else:
        genre = clicked_genre['points'][0]['customdata'][0]  # How do we get multi-select?

        if option_slct == 1:
            minVotes = 25000
            rank_count = dfMovies.copy()
            mask_count = (rank_count['year'].astype(int) >= low) & (rank_count['year'].astype(int) <= high) & (
                    rank_count['genres'].astype(str) == genre)

        else:
            minVotes = 10000
            rank_count = dfShows.copy()
            mask_count = filterTVyears(rank_count, low, high, genre=genre)

        if rateorpop == 1:
            topRank = rank_count[mask_count].nlargest(10, 'averageRating')
            topRank.sort_values(by='averageRating', inplace=True)

            fig = px.bar(
                topRank, y='primaryTitle', x='averageRating', color='numVotes', orientation='h',
                labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                            primaryTitle="Title", genres='Genre', runtimeMinutes='Runtime in Minutes',
                            year='Year(s)'),
                title='Top 10 {}, minimum {} votes'.format(genre, minVotes),
                hover_name=topRank.primaryTitle,
                hover_data=['averageRating', 'numVotes', 'runtimeMinutes', 'year'],
                color_continuous_scale='pinkyl',
                template='plotly_dark'
            )
        else:
            topPop = rank_count[mask_count].nlargest(10, 'numVotes')
            topPop.sort_values(by='numVotes', inplace=True)

            fig = px.bar(
                topPop, y='primaryTitle', x='numVotes', color='averageRating', orientation='h',
                labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                            primaryTitle="Title", genres='Genre', runtimeMinutes='Runtime in Minutes',
                            year='Year(s)'),
                title='Top 10 {}, minimum {} votes'.format(genre, minVotes),
                hover_name=topPop.primaryTitle,
                hover_data=['averageRating', 'numVotes', 'runtimeMinutes', 'year'],
                color_continuous_scale='pinkyl',
                template='plotly_dark'
            )

    return fig


# Modal (info button)
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# DASH -> Main page
app.layout = dbc.Container(
    [
        dbc.Row([
             #dbc.Col([
             #  html.Img(src='https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/IMDB_Logo_2016.svg/2560px-IMDB_Logo_2016.svg.png',
             #            style={'display': 'inline-block',
             #                   'height':'auto',
             #                   'object-fit':'contain',
             #                   'vertical-align':'center'}),
             #    ],width = 3, align = 'start'),
             dbc.Col([
                html.H1("IMDb Data Explorer", className='text-center', style={
                    #'text_align': 'center',
                    'color': '#e0b416',
                    })
                 ],width = 12),
        ]),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(id='TVorMovie',
                             options=[
                                 {'label': 'Movies', 'value': 1},  # 1 = movies
                                 {'label': 'TV Shows', 'value': 2}],
                             clearable=False,
                             style={'backgroundColor':'#e0b416', 'border-color':'#e0b416',
                                    'color':'#FFFFFF'},
                             value=1, className = 'select_box'),
                ], width=3, align='center', style={'color':'white'}),
            dbc.Col([
                dcc.Dropdown(id='RateOrPop',
                             options=[
                                 {'label': 'Focus: Ratings', 'value': 1},  # 1 = movies
                                 {'label': 'Focus: Popularity', 'value': 2}],
                             clearable=False,
                             style={
                                    'background-color':'#e0b416',
                                    'border-color':'#e0b416','color':'#FFFFFF'
                                    },
                             value=1, className = 'select_box'),
                ], width=3, align='center'),
            dbc.Col([
                dbc.Button("Info", id="open", n_clicks=0, color='warning', active = True,
                           className='me-1', style={'height':'25%'}),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Page Information", style={'color':'#e0b416'}), close_button=False,
                                        style={'background-color':'black', 'color':'white'}),
                        dbc.ModalBody('Welcome to the IMDb Visualization! '
                                      '\n'
                                      '\nAll data used is sourced from IMDb\'s datasets. '
                                      'Please note that I am not affiliated with IMDb. '
                                      'Datasets used were updated in February 2022. '
                                      'Entries were filtered by a minimum number of '
                                      'votes in accordance with IMDb ranking lists. '
                                      '\nAt the top you may switch between looking at movies or TV shows, and whether '
                                      'you want to focus on the ratings, or the number of ratings (popularity).'
                                      ' You may open this dialogue again by clicking the info button on the top right.'
                                      '\n'
                                      '\n'
                                      'Bubble Graph:'
                                      '\nUse the main bubble graph to modify the page. The size of the bubbles '
                                      'represent the number of movies or shows released from that genre within the '
                                      'given year range. Click a bubble to filter the page '
                                      'by the chosen genre. Click it again to unselect. Click genres on the legend'
                                      ' on the right to hide bubbles, or double click to isolate. The figure Can be zoomed into.'
                                      '\n'
                                      '\n'
                                      'KDE chart:'
                                      '\nThis chart shows the estimated distribution based on. It can essentially be looked'
                                      ' at as a vertical, double-sided approximate histogram. The individual points are '
                                      'shown on its left.'
                                      '\n'
                                      '\n'
                                      'Runtime and Bar:'
                                      '\nFurther below is the runtime histogram with a rating/votes trendline, and a bar '
                                      'chart of the highest ranking filtered movies or TV shows.'
                                      '', style={'background-color':'black', 'color':'white'}),
                        dbc.ModalFooter(
                            dbc.Button(
                                'Close', id="close", className='ms-auto', color='dark', n_clicks=0,
                                style={'background-color':'#e0b416', 'color':'black'}
                            ), style={'background-color':'black', 'color':'white'}
                        ),
                    ],
                    id='modal',
                    is_open=True,
                    size='lg',
                    autofocus= True,
                    style={'white-space': 'pre-line'}
                ),
                ], width ={'size':3, 'offset':3}, align='center'),
            ], style={'padding':2}),

        html.Div(id='output_container', children=[], style={'backgroundColor': '#111111'}),

        dcc.RangeSlider(1906, 2022, 1, id='year_slider', updatemode='drag',
                        tooltip={"placement": "bottom", "always_visible": True},
                        marks={i: '{}'.format(i) for i in range(1906, 2022, 12)},
                        value=[1906, 2022],
                        ),

        dbc.Row([
           dbc.Col([
               dcc.Graph(id='bubble', figure={}, config={'doubleClick': 'reset', 'showTips': True}, selectedData=None),
           ],xs=12,lg=8),
           dbc.Col([
               dcc.Graph(id='violin', figure={}, config={'doubleClick': 'reset', 'showTips': True},
                         className='six columns',  #style={'width': '25%'}
                         ),
           ],xs=12,lg=4),
        ]),
        html.Br(),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id='runTime', figure={}, config={'doubleClick': 'reset', 'showTips': True},),
                ],xs=12,lg=4,xl=5),
            dbc.Col([
                dcc.Graph(id='toprank', figure={}, config={'doubleClick': 'reset', 'showTips': True},),
            ],xs=12,lg=8, xl=7),
        ]),
    ],
    style={'backgroundColor': '#111111',
           'color': '#FFFFFF'}
)

if __name__ == '__main__':
    app.run_server(debug=True)
