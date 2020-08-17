import os
import pickle
import pandas as pd
import plotly.graph_objects as go
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import widgets

#Html builder
import dash
from dash.dependencies import Input, Output
import dash_table
import dash_core_components as dcc
import dash_html_components as html
#basic webserver
import flask


indicators_file = 'ego_state_indicators.obj'
sentences_file = 'sentences.obj'

with open(indicators_file, 'rb') as f:
    indicators = pickle.loads(f.read())
 
with open(sentences_file, 'rb') as f:
    sentences = pickle.loads(f.read())
    
    
total_indicators = sum(i.count() for i in indicators)

indicators = sorted(indicators, key=lambda i: i.idx)

df = pd.DataFrame(None, columns = ['ID', 'Channel', 'Label', 'Indicator'])

for indicator in indicators:
    for k in indicator.adult:
        d = {'ID': indicator.idx, 'Channel':indicator.channel, 'Label':'Adult', 'Indicator':k}
        if indicator.channel == 'Text':
            d['Content'] = sentences[indicator.idx]
        df = df.append(d, ignore_index=True)
            
    for k in indicator.child:
        d = {'ID': indicator.idx, 'Channel':indicator.channel, 'Label':'Child', 'Indicator':k}
        if indicator.channel == 'Text':
            d['Content'] = sentences[indicator.idx]
        df = df.append(d, ignore_index=True)
    for k in indicator.parent:
        d = {'ID': indicator.idx, 'Channel':indicator.channel, 'Label':'Parent', 'Indicator':k}
        if indicator.channel == 'Text':
            d['Content'] = sentences[indicator.idx]
        df = df.append(d, ignore_index=True)
    

segmentationDropdown = dcc.Dropdown(
    options= list(map(lambda i: {'label':i, 'value':i}, [0, 1,2,4,8,16])),
    value=0,
    id='segmentation',
    searchable=False
)
table = dash_table.DataTable( id='data-table',
    columns=[{'name':'Frame', 'id':'ID'}] + [{"name": i, "id": i} for i in ['Channel', 'Label', 'Indicator', 'Content']],
    data=pd.DataFrame().to_dict('records'),
    row_selectable='single',
)  

segTable = dash_table.DataTable( id='segmentation-table',
    columns=[{"name": i, "id": i} for i in ['Second', 'Adult', 'Child','Parent', 'Transcript']],
    data=pd.DataFrame({}).to_dict('records'),
    row_selectable='single',
)  

video = html.Video(id='media', 
            src="http://localhost:8050/demo/speaker2.mp4", 
            width=480, 
            controls=True)

hiddenVideo = html.Video(id='hidden-media', 
            src="http://localhost:8050/demo/speaker2.mp4", 
            width=480)



#origin.observe(response, names="value")


#interact(response, x=widgets.IntSlider(min=-10, max=30, step=1, value=10));
#container = widgets.HBox([origin])
#display(interactive(response, seconds=segmentationWidget))
server = flask.Flask(__name__)


@server.route('/demo/<path:path>')
def serve_demo(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'demo'), path)

@server.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'static'), path)


app = dash.Dash(__name__, server=server)


loadSpinner = html.Div([html.Div([html.Div(), html.Div(), html.Div()], className='lds-spinner'),'Loading...'], className='lds-container')


app.layout = html.Div([html.Div(
    [
        html.Div([
            html.H1(["Eric Berne's Parent, Adult, Child classification"], className='font-bold text-1xl'),
        ], className='header'),
        html.Div([
            html.Label(['Video Clip']), 
            video
        ], className='left'),
        html.Div([
            html.Label(['Indicators']),
            html.Div([table], id='indicators'),
            loadSpinner
        ], className='right'),
        html.Div([
            html.Label(["Segmentation (seconds)", segmentationDropdown]),
            html.Div([segTable], id='segments')
            
        ], className='footer')
            
    ], className='grid-container'),
        html.Canvas([], id='hidden-canvas'),
        hiddenVideo
])



@app.callback(
        Output('media','src'),
        [Input('segmentation', 'value'), Input('segmentation-table','selected_rows')])
def updateMediaUrl(seg, row):
    url = '/demo/speaker2.mp4'
    if seg == 0 or seg is None or row is None:
        return url
    return url + '#t={0},{1}'.format(row[0]*seg, (row[0]+1)*seg)

@app.callback(
        Output('hidden-media','src'),
        [Input('segmentation', 'value'), Input('segmentation-table','selected_rows')])
def updateMediaUrl(seg, row):
    url = '/demo/speaker2.mp4'
    if seg == 0 or seg is None or row is None:
        return url
    return url + '#t={0},{1}'.format(row[0]*seg, (row[0]+1)*seg)

@app.callback(
    Output('segments', 'children'),
    [Input('segmentation', 'value')])
def updateSegmentation(seconds):
    segmentationDropdown.value = None
    output = {}
    if seconds is None or seconds == 0:

        for i in range(len(indicators)):
            indicator = indicators[i]
            idx_prior = 'start'
            idx = 'end'
            segment = output.get(idx,{'Second':"{0} to {1}".format(idx_prior, idx), 'Adult':0, 'Child':0, 'Parent':0, 'Transcript':''})


            segment['Adult'] += len(indicator.adult)
            segment['Child'] += len(indicator.child)
            segment['Parent'] += len(indicator.parent)

            if indicator.channel == 'Text':
                segment['Transcript'] += ' ' + sentences[indicator.idx]

            output[idx] = segment

        df2 = pd.DataFrame(None, columns=['Second', 'Adult', 'Child','Parent', 'Transcript'])
        sorted_keys = list(output.keys())
        sorted_keys.sort()

        #Calculate frequentist's probability
        for i in sorted_keys:
            o = output[i]
            t = o['Adult'] + o['Child'] + o['Parent']

            o['Adult'] = o['Adult']/t
            o['Child'] = o['Child']/t
            o['Parent'] = o['Parent']/t

            df2 = df2.append(output[i], ignore_index=True)


        return [dash_table.DataTable(id='segmentation-table', 
                   columns=[{"name": i, "id": i} for i in ['Second', 'Adult', 'Child', 'Parent', 'Transcript']],
                   style_cell_conditional=[
                        {'if': {'column_id': 'Transcript'},
                            'text-overflow':'ellipsis'},
                   ],
                   data=df2.to_dict('records'),
                   row_selectable=False
               )]  


    else:
        #assume 25 frames per second
        segment_denom = seconds*25
        for i in range(len(indicators)):
            indicator = indicators[i]
            idx_prior = int(indicator.idx/segment_denom) *seconds
            idx = (int(indicator.idx/segment_denom)+1) *seconds
            segment = output.get(idx,{'Second':"{0} to {1}".format(idx_prior, idx), 'Adult':0, 'Child':0, 'Parent':0, 'Transcript':''})


            segment['Adult'] += len(indicator.adult)
            segment['Child'] += len(indicator.child)
            segment['Parent'] += len(indicator.parent)
            if indicator.channel == 'Text':
              segment['Transcript'] += ' ' + sentences[indicator.idx]

            output[idx] = segment

        df2 = pd.DataFrame(None, columns=['Second', 'Adult', 'Child','Parent', 'Transcript'])
        sorted_keys = list(output.keys())
        sorted_keys.sort()

        #Calculate frequentist's probability
        for i in sorted_keys:
            o = output[i]
            t = o['Adult'] + o['Child'] + o['Parent']

            o['Adult'] = o['Adult']/t
            o['Child'] = o['Child']/t
            o['Parent'] = o['Parent']/t

            df2 = df2.append(output[i], ignore_index=True)

        return [dash_table.DataTable(id='segmentation-table', 
                   columns=[{"name": i, "id": i} for i in ['Second', 'Adult', 'Child', 'Parent', 'Transcript']],
                   style_cell_conditional=[
                        {'if': {'column_id': 'Transcript'},
                            'text-overflow':'ellipsis'},
                   ],
                   data=df2.to_dict('records'),
                   row_selectable='single'
               )]  

        
        
@app.callback(Output('indicators','children'),[Input('segmentation-table','selected_rows'), Input('segmentation','value') ])
def updateIndicators(id, seg):
    if id is None or seg is None or seg == 0:
        return [dash_table.DataTable(id='data-table', columns=[{'name':'Frame', 'id':'ID'}] + [{"name": i, "id": i} for i in ['Channel', 'Label', 'Indicator', 'Content']],
            data=df.to_dict('records'),
            row_selectable=False
        )]  
    return [dash_table.DataTable(columns=[{'name':'Frame', 'id':'ID'}] + [{"name": i, "id": i} for i in ['Channel', 'Label', 'Indicator', 'Content']],
            data=df.loc[(df['ID'] >= (id[0]*seg*25)) & (df['ID'] < ((id[0]+1)*seg*25))].to_dict('records')
        )]  

    


app.run_server()
