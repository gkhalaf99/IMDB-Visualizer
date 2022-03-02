import pandas as pd
# import ipywidgets as widgets
# import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# from jupyter_dash import JupyterDash
from dash import dcc, html, Dash
from dash.dependencies import Output, Input

# IMPORTING AND MERGING THE DATA
dfBasics = pd.read_csv('Downloads/title.basics.tsv', sep='\t', header=0)
# dfRegion = pd.read_csv('Downloads/title.akas.tsv', sep='\t', header=0)
dfRatings = pd.read_csv('Downloads/title.ratings.tsv', sep='\t', header=0)

dfTotal = pd.merge(dfBasics, dfRatings, on='tconst')

# LOADING MOVIES
dfMovies = dfTotal.loc[dfTotal['titleType'] == 'movie']
dfMovies.rename({'startYear': 'year'}, axis=1, inplace=True)
dfMovies = dfMovies[dfMovies.runtimeMinutes != '\\N']
dfMovies = dfMovies[dfMovies.genres != '\\N']
dfMovies = dfMovies[dfMovies.year != '\\N']
dfMovies = dfMovies[dfMovies['numVotes'].astype(int) >= 25000]
# Fixing movies so that the genres are split into different rows
dfMovies.drop(['originalTitle'], axis=1, inplace=True)
dfMovies.drop(['endYear'], axis=1, inplace=True)
dfMovies.set_index(['tconst'], inplace=True)

dfMovies['genres'] = dfMovies['genres'].str.split(',')
dfMovies = dfMovies.explode('genres')
dfMovies = dfMovies[dfMovies.genres != 'Game-Show']

# LOADING TV SHOWS
dfShows = dfTotal.loc[dfTotal['titleType'].isin(['tvSeries', 'tvMiniSeries'])]
dfShows = dfShows[dfShows.runtimeMinutes != '\\N']
dfShows = dfShows[dfShows.genres != '\\N']
dfShows = dfShows[dfShows.startYear != '\\N']
dfShows = dfShows[dfShows['numVotes'].astype(int) >= 10000]
dfShows.endYear.replace('\\N', 2022, inplace=True)  # REMOVE IF I WANNA SCRAP THE CHANGE OF \N to 2022
# Fixing the TV Shows database so that genres are split into different rows
dfShows.drop(['originalTitle'], axis=1, inplace=True)
dfShows.set_index(['tconst'], inplace=True)

dfShows['genres'] = dfShows['genres'].str.split(',')
dfShows = dfShows.explode('genres')

# DASH -> Main page

external_stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheet)  # JupyterDash for here
server = app.server

app.layout = html.Div([

    html.H1("IMDb Data Explorer", style={'text_align': 'center'}),

    dcc.Dropdown(id='TVorMovie',
                 options=[
                     {'label': 'Movies', 'value': 1},  # 1 = movies
                     {'label': 'TV Shows', 'value': 2}],
                 style={'width': '40%'},
                 value=1),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.RangeSlider(1906, 2022, 1, id='year_slider', updatemode='drag',
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks={i: '{}'.format(i) for i in range(1906, 2022, 4)},
                    value=[1906, 2022]),

    dcc.Graph(id='bubble', figure={}, config={'doubleClick': 'reset', 'showTips': True}, selectedData=None),

    html.Br(),

    html.Div([
        dcc.Graph(id='violin', figure={}, config={'doubleClick': 'reset', 'showTips': True},
                  className='six columns', style={'width': '40%'}),

        dcc.Graph(id='runTime', figure={}, config={'doubleClick': 'reset', 'showTips': True},
                  className='six columns', style={'width': '40%'}),

        dcc.Graph(id='toprank', figure={}, config={'doubleClick': 'reset', 'showTips': True},
                  className='six columns', style={'width': '40%'})

    ], className='row')  # Possible to add a scroll bar?

])

del app.config._read_only["requests_pathname_prefix"]
app.run_server(mode='external')


# BUBBLE GRAPH UPDATING
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='bubble', component_property='figure')],
    [Input(component_id='TVorMovie', component_property='value'),
     Input(component_id='year_slider', component_property='value')]
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
        mask_count = (bubble_count['startYear'].astype(int) >= low) & (bubble_count['endYear'].astype(int) <= high)

    fig = px.scatter(
        bubble_count[mask_count].groupby('genres').mean(), x='numVotes', y='averageRating',
        size=bubble_count[mask_count].groupby('genres')['numVotes'].count(),
        color=bubble_count[mask_count].groupby('genres').genres.first(),
        hover_name=bubble_count[mask_count].groupby('genres').genres.first(),
        custom_data=[bubble_count[mask_count].groupby('genres').genres.first()],
        labels=dict(numVotes="Average Number of Votes", averageRating="Average Rating out of 10",
                    size="Total Count", color='Genre'),
        log_x=False, size_max=65,
        title='Number of votes vs Average Rating out of 10 Filtered by Genre and/or Year(s)'
        # template = 'plotly_dark'
    )

    fig.update_layout(clickmode='event+select')

    return container, fig


# VIOLIN GRAPH UPDATING
@app.callback(
    Output(component_id='violin', component_property='figure'),
    [Input(component_id='TVorMovie', component_property='value'),
     Input(component_id='year_slider', component_property='value'),
     Input(component_id='bubble', component_property='selectedData')]
)
def update_violin(option_slct, year_slct, clicked_genre):
    low, high = year_slct

    if clicked_genre is None:

        if option_slct == 1:

            violin_count = dfMovies.copy()
            mask_count = (violin_count['year'].astype(int) >= low) & (violin_count['year'].astype(int) <= high)

            violin_av = violin_count[mask_count].groupby('genres').mean()
            violin_av.reset_index(inplace=True)
        else:

            violin_count = dfShows.copy()
            mask_count = (violin_count['startYear'].astype(int) >= low) & (violin_count['endYear'].astype(int) <= high)

            violin_av = violin_count[mask_count].groupby('genres').mean()
            violin_av.reset_index(inplace=True)

        fig = px.violin(
            violin_av, y='averageRating', box=True, points='all',
            labels=dict(numVotes="Average Number of Votes", averageRating="Average Rating out of 10",
                        size="Total Count", genres='Genre'), title='KDE of genres in the given years',
            hover_name=violin_av.genres,
            hover_data=['averageRating', 'numVotes'],
            # template = 'plotly_dark'
        )
    else:

        genre = clicked_genre['points'][0]['customdata'][0]  # How do we get multi-select?

        if option_slct == 1:

            violin_count = dfMovies.copy()
            mask_count = (violin_count['year'].astype(int) >= low) & (violin_count['year'].astype(int) <= high) & (
                        violin_count['genres'].astype(str) == genre)
        else:

            violin_count = dfShows.copy()
            mask_count = (violin_count['startYear'].astype(int) >= low) & (
                        violin_count['endYear'].astype(int) <= high) & (violin_count['genres'].astype(str) == genre)

        fig = px.violin(
            violin_count[mask_count], y='averageRating', box=True, points='all',
            labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                        size="Total Count"),
            title='KDE of the {} genre'.format(genre),
            hover_name=violin_count[mask_count].primaryTitle,
            hover_data=['averageRating', 'numVotes'],
            # template = 'plotly_dark'
        )

    return fig


# RUNTIME GRAPH UPDATING   #TOGGLE BETWEEN MOST VOTES AND TOP RANKS

@app.callback(
    Output(component_id='runTime', component_property='figure'),
    [Input(component_id='TVorMovie', component_property='value'),
     Input(component_id='year_slider', component_property='value'),
     Input(component_id='bubble', component_property='selectedData')]
)
def update_RUN(option_slct, year_slct, clicked_genre):
    low, high = year_slct

    if clicked_genre is None:

        if option_slct == 1:

            run_count = dfMovies.copy()
            mask_count = (run_count['year'].astype(int) >= low) & (run_count['year'].astype(int) <= high)

        else:

            run_count = dfShows.copy()
            mask_count = (run_count['startYear'].astype(int) >= low) & (run_count['endYear'].astype(int) <= high)


    else:

        genre = clicked_genre['points'][0]['customdata'][0]  # How do we get multi-select?

        if option_slct == 1:

            run_count = dfMovies.copy()
            mask_count = (run_count['year'].astype(int) >= low) & (run_count['year'].astype(int) <= high) & (
                        run_count['genres'].astype(str) == genre)

        else:

            run_count = dfShows.copy()
            mask_count = (run_count['startYear'].astype(int) >= low) & (run_count['endYear'].astype(int) <= high) & (
                        run_count['genres'].astype(str) == genre)

    run_count.runtimeMinutes = run_count.runtimeMinutes.astype(int)
    run_count2 = run_count[mask_count].sort_values(by='runtimeMinutes', ascending=True)
    run_count = run_count2.groupby(['primaryTitle', 'runtimeMinutes'], as_index=False).mean()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # create two independent figures with px.line each containing data from multiple columns
    fig0 = px.histogram(run_count2, x='runtimeMinutes')
    fig0.update_traces(yaxis='y1')

    fig1 = px.line(run_count.groupby('runtimeMinutes', as_index=False).mean(),
                   x='runtimeMinutes', y='averageRating', render_mode="webgl", )
    fig1.update_traces(line=dict(color="#ff5733"), yaxis='y2')

    fig.add_traces(fig0.data + fig1.data)
    fig.layout.xaxis.title = "Runtime in Minutes"
    fig.layout.yaxis.title = "Count"
    # subfig.layout.yaxis2.type="log"
    fig.layout.yaxis2.title = "Average Rating out of 10"
    # recoloring is necessary otherwise lines from fig und fig2 would share each color

    return fig


# RANKINGS GRAPH UPDATING   #TOGGLE BETWEEN MOST VOTES AND TOP RANKS

@app.callback(
    Output(component_id='toprank', component_property='figure'),
    [Input(component_id='TVorMovie', component_property='value'),
     Input(component_id='year_slider', component_property='value'),
     Input(component_id='bubble', component_property='selectedData')]
)
def update_ranks(option_slct, year_slct, clicked_genre):
    low, high = year_slct

    if clicked_genre is None:

        if option_slct == 1:

            minVotes = 25000
            rank_count = dfMovies.copy()
            mask_count = (rank_count['year'].astype(int) >= low) & (rank_count['year'].astype(int) <= high)

        else:

            minVotes = 10000
            rank_count = dfShows.copy()
            mask_count = (rank_count['startYear'].astype(int) >= low) & (rank_count['endYear'].astype(int) <= high)

        topRank = rank_count[mask_count]
        topRank = topRank.groupby(['primaryTitle', 'numVotes'], as_index=False).agg(
            {'averageRating': 'first', 'genres': lambda x: ','.join(x.astype(str))})

        topRank = topRank.nlargest(10, 'averageRating')
        topRank.sort_values(by='averageRating', inplace=True)

        fig = px.bar(
            topRank, y='primaryTitle', x='averageRating', color='numVotes', orientation='h',
            labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                        primaryTitle="Title", genres='Genre'), title='Top 10, minimum {} votes'.format(minVotes),
            hover_name=topRank.primaryTitle,
            hover_data=['averageRating', 'numVotes', 'genres'],
            # template = 'plotly_dark'
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
            mask_count = (rank_count['startYear'].astype(int) >= low) & (rank_count['endYear'].astype(int) <= high) & (
                        rank_count['genres'].astype(str) == genre)

        topRank = rank_count[mask_count].nlargest(10, 'averageRating')
        topRank.sort_values(by='averageRating', inplace=True)

        fig = px.bar(
            topRank, y='primaryTitle', x='averageRating', color='numVotes', orientation='h',
            labels=dict(numVotes="Total Number of Votes", averageRating="Average Rating out of 10",
                        primaryTitle="Title", genres='Genre'),
            title='Top 10 {}, minimum {} votes'.format(genre, minVotes),
            hover_name=topRank.primaryTitle,
            hover_data=['averageRating', 'numVotes'],
            # template = 'plotly_dark'
        )

    return fig
