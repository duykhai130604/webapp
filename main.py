import os
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('filetonghop.csv')
new_df = pd.read_csv('tonghop.csv')
df['Date'] = pd.to_datetime(df['Date'])
new_df['Date'] = pd.to_datetime(new_df['Date'])
filtered_df = new_df[new_df['Date'] >= '2020-01-01']
df_growth = pd.DataFrame(columns=['Company', 'Percent_Growth_Rate'])
companies = filtered_df['Company'].unique()
for company in companies:
    company_df = filtered_df[filtered_df['Company'] == company]
    adj_close_dau = company_df.iloc[0]['Adj Close']
    adj_close_cuoi = company_df.iloc[-1]['Adj Close']
    growth_rate = ((adj_close_cuoi - adj_close_dau) / adj_close_dau) * 100
    temp_df = pd.DataFrame({'Company': [company], 'Percent_Growth_Rate': [growth_rate]})
    df_growth = pd.concat([df_growth, temp_df], ignore_index=True)
df_growth.dropna(axis=1, how='all', inplace=True)
results = {}
available_colors = [
    'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 
    'cyan', 'magenta', 'yellow'
]
for company in companies:
    company_df = df[df['Company'] == company].sort_values(by='Date')
    features = ['Open', 'High', 'Low', 'Volume']
    target = 'Close'
    X = company_df[features]
    y = company_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    model = ElasticNet(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 1 - mape
    results[company] = {
        'RMSE': rmse, 
        'R²': r2,
        'MAPE': mape,
        'Accuracy': accuracy,
        'Predicted': y_pred, 
        'Actual': y_test.values, 
        'Dates': company_df['Date'][len(y_train):].values
    }
result_df = pd.DataFrame(columns=['Company', 'Total Values', 'Outliers', 'Non-Outliers'])
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((data < lower_bound) | (data > upper_bound))
for company in new_df['Company'].unique():
    company_df = new_df[new_df['Company'] == company]
    outliers = detect_outliers_iqr(company_df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
    total_values = company_df.shape[0]
    outliers_count = outliers.sum().sum() 
    non_outliers_count = total_values - outliers_count
    result_df = pd.concat([result_df, pd.DataFrame({
        'Company': [company],
        'Total Values': [total_values],
        'Outliers': [outliers_count],
        'Non-Outliers': [non_outliers_count]
    })], ignore_index=True)
fig_growth = go.Figure(data=[
    go.Bar(x=df_growth['Company'], y=df_growth['Percent_Growth_Rate'],
           marker=dict(color='skyblue', line=dict(color='black', width=1)),
           text=[f"{x:.2f}%" for x in df_growth['Percent_Growth_Rate']],
           textposition='auto')
])
fig_growth.update_layout(
    title='Price Growth Of Fast Food Companies (Since 01/01/2020)',
    xaxis_title='Company',
    yaxis_title='Growth Rate (%)',
    xaxis_tickangle=-45,
    template='plotly_white'
)
adj_close_data = pd.DataFrame()
for company in companies:
    company_df = filtered_df[filtered_df['Company'] == company]
    adj_close_data[company] = company_df.set_index('Date')['Adj Close']
correlation_matrix = adj_close_data.corr()
fig_heatmap = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='Reds', 
    text=correlation_matrix.values,
    texttemplate='%{text:.2f}',
    zmin=-1, zmax=1
))
fig_heatmap.update_layout(
    title='Matrix Correlation',
    xaxis_title='Company',
    yaxis_title='Company',
    autosize=False,
    width=800, 
    height=800 
)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='FAST FOOD INDUSTRY STOCK TREND AND PERFORMANCE ANALYSIS', children=[
            html.H1("Dataset by Yahoo Finance", style={'text-align': 'center'}),
            dcc.Dropdown(
                id='company_dropdown',
                options=[{'label': company, 'value': company} for company in companies],
                value=companies[0],
                style={'width': '40%', 'margin': 'auto'}
            ),
            html.Div(id='filtered_table'),
            html.H1("Data Re-processing", style={'text-align': 'center', 'margin-top': '50px'}),
            dcc.Dropdown(
                id='company-dropdown',
                options=[{'label': company, 'value': company} for company in result_df['Company']],
                value=result_df['Company'][0],
                style={'width': '30%', 'margin': 'auto'}
            ),
            dcc.Graph(id='indicator-graphic'),
            html.H1("Price Growth Of Fast Food Companies", style={'text-align': 'center', 'margin-top': '50px'}),
            dcc.Graph(figure=fig_growth, style={'margin': 'auto', 'width': '80%'}),
            html.H1("Correlation Chart Between Fast Food Companies", style={'text-align': 'center', 'margin-top': '50px'}),
            dcc.Graph(figure=fig_heatmap, style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin': 'auto', 'width': '80%'}),
            html.H1("Stock Price Data by Skywalker", style={'text-align': 'center', 'margin-top': '50px'}),
            html.Div([
                html.Div("Choose ticker", style={'text-align': 'center'}),
                dcc.Dropdown(
                    id='company_selector',
                    options=[{'label': company, 'value': company} for company in companies],
                    value=[],
                    multi=True,
                    style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "40%"}
                ),
                html.Div("Choose value", style={'text-align': 'center', 'margin-top': '20px'}),
                dcc.Dropdown(
                    id='value_selector',
                    options=[
                        {'label': 'Open', 'value': 'Open'},
                        {'label': 'High', 'value': 'High'},
                        {'label': 'Low', 'value': 'Low'},
                        {'label': 'Close', 'value': 'Close'},
                        {'label': 'Adj Close', 'value': 'Adj Close'},
                        {'label': 'Volume', 'value': 'Volume'},
                        {'label': 'SMA_20', 'value': 'SMA_20'},
                        {'label': 'SMA_50', 'value': 'SMA_50'},
                        {'label': 'SMA_200', 'value': 'SMA_200'}
                    ],
                    value=[],
                    multi=True,
                    style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "40%"}
                ),
                html.Br(),
                html.Div(
                                        [
                        html.Span('Date', style={"textAlign": "center", 'margin-right': '10px'}),
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            start_date=df['Date'].min(),
                            end_date=df['Date'].max(),
                            display_format='YYYY-MM-DD',
                            style={"display": "block", "margin-left": "auto", "margin-right": "auto"}
                        ),
                        html.Br(),
                        dcc.Checklist(
                            id='prediction-lines',
                            options=[
                                {'label': 'Actual', 'value': 'Actual'},
                                {'label': 'Prediction', 'value': 'Predicted'}
                            ],
                            value=[],
                            style={"display": "flex", "justify-content": "center", 'gap': '10px', 'textAlign': 'center'}
                        ),
                    ], style={'textAlign': 'center'}
                ),
                dcc.Graph(id='stock_graph', style={'margin-top': '20px'})
            ])
        ])
    ])
])
@app.callback(
    Output('filtered_table', 'children'),
    [Input('company_dropdown', 'value')]
)
def update_table(company):
    filtered_df = new_df[new_df['Company'] == company]
    filtered_df = filtered_df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].head(15)
    return html.Div([
        html.H3(f"Filtered data for {company}", style={'text-align': 'center'}),
        html.Div(
            dcc.Markdown(f"```\n{filtered_df.to_string(index=False)}\n```"),
            style={'text-align': 'center', 'margin': 'auto', 'width': '50%'}
        )
    ])
@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('company-dropdown', 'value')]
)
def update_indicator_graph(selected_company):
    filtered_df = result_df[result_df['Company'] == selected_company]
    if filtered_df.empty:
        return {
            'data': [],
            'layout': go.Layout(
                title=f'Indicators for {selected_company}',
                yaxis={'title': 'Count'},
                barmode='group'
            )
        }
    traces = []
    traces.append(go.Bar(
        x=['Total Values', 'Outliers', 'Non-Outliers'],
        y=[
            filtered_df['Total Values'].values[0], 
            filtered_df['Outliers'].values[0], 
            filtered_df['Non-Outliers'].values[0]
        ],
        name=selected_company
    ))
    return {
        'data': traces,
        'layout': go.Layout(
            title=f'Indicators for {selected_company}',
            yaxis={'title': 'Count'},
            barmode='group'
        )
    }
@app.callback(
    Output('stock_graph', 'figure'),
    [Input('company_selector', 'value'),
     Input('value_selector', 'value'),
     Input('prediction-lines', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_stock_graph(selected_companies, selected_values, selected_lines, start_date, end_date):
    if not selected_companies:
        return go.Figure()
    filtered_df = df[(df['Company'].isin(selected_companies)) & 
                     (df['Date'] >= start_date) & 
                     (df['Date'] <= end_date)]
    fig = go.Figure()
    used_colors = []
    for value in selected_values:
        for company in selected_companies:
            color = random.choice([c for c in available_colors if c not in used_colors])
            used_colors.append(color)
            company_df = filtered_df[filtered_df['Company'] == company]
            fig.add_trace(go.Scatter(
                x=company_df['Date'], 
                y=company_df[value], 
                mode='lines', 
                name=f'{company} - {value}',
                line=dict(color=color)
            ))
    if 'Actual' in selected_lines or 'Predicted' in selected_lines:
        for company in selected_companies:
            data = results[company]
            if 'Actual' in selected_lines:
                color = random.choice([c for c in available_colors if c not in used_colors])
                used_colors.append(color)
                fig.add_trace(go.Scatter(
                    x=data['Dates'], 
                    y=data['Actual'], 
                    mode='lines', 
                    name=f'{company} - Actual',
                    line=dict(color=color)
                ))
            if 'Predicted' in selected_lines:
                color = random.choice([c for c in available_colors if c not in used_colors])
                used_colors.append(color)
                fig.add_trace(go.Scatter(
                    x=data['Dates'], 
                    y=data['Predicted'], 
                    mode='lines', 
                    name=f'{company} - Prediction',
                    line=dict(color=color)
                ))
                fig.add_annotation(
                    xref='paper', yref='paper',
                    x=0.5, y=1.15,
                    xanchor='center', yanchor='top',
                    text = (f"RMSE: {results[company]['RMSE']:.2f}, "
                            f"MAPE: {results[company]['MAPE']:.2f}, "
                            f"R²: {results[company]['R²']:.2f}, "
                            f"Accuracy: {results[company]['Accuracy']:.2f}"),
                    showarrow=False,
                    font=dict(size=14)
                )
    fig.update_layout(
        title=f"{', '.join(selected_values)} over Time",
        template='plotly_dark'
    )
    return fig
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=True)