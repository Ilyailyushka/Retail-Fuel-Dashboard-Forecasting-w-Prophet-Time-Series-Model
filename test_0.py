from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from prophet import Prophet

# Read CSV files
read_sales = pd.read_csv("/Users/mca8461/Documents/pet_project/data/retail_data_analytics/sales data-set.csv")
read_features = pd.read_csv("/Users/mca8461/Documents/pet_project/data/retail_data_analytics/Features data set.csv")
read_stores = pd.read_csv("/Users/mca8461/Documents/pet_project/data/retail_data_analytics/stores data-set.csv")

# Merge datasets and preprocess date columns
df = read_sales.merge(read_features, left_on=['Store', 'Date'], right_on=['Store', 'Date']).merge(read_stores, left_on='Store', right_on='Store')
df['Time_Format'] = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
df = df.groupby(['Store', 'Time_Format']).sum().reset_index()

# Set up data for Prophet model
df['y'] = df.Weekly_Sales
df['ds'] = df.Time_Format
df_valid = df[df.Time_Format.apply(lambda x: x.date()) >= datetime.date(year=2012, month=4, day=1)]
df_train = df[df.Time_Format.apply(lambda x: x.date()) < datetime.date(year=2012, month=4, day=1)]

# Define Plotly template
TEMPLATE = "plotly_white"

# Initialize Dash app
app = Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1(children='Retail Fuel Dashboard Forecasting', style={'textAlign':'center'}),
    dcc.Dropdown(df.Store.unique(), 1, id='dropdown-selection'),
    html.Button('Create forecast', id='fcst-btn', n_clicks=0, ),
    html.Div(id='output_text'),
    dcc.Graph(id='graph-content')
])

# Define callback function for updating graph
@callback(
    Output('graph-content', 'figure'),
    State('dropdown-selection', 'value'),
    Input('fcst-btn', 'n_clicks')
)
def plot_nyiso_load_(value, value_btn):
    # Subset data based on selected store
    dff_train = df_train[df_train.Store == value]
    dff_valid = df_valid[df_valid.Store == value]

    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(dff_train)

    # Make predictions for the validation set
    predict = model.predict(dff_valid)

    # Create Plotly figure with actual sales, holiday flag, and forecast
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dff = df[df.Store == value]
    
    # Actual Weekly Sales
    fig.add_trace(
        go.Scatter(
            x=dff.Time_Format,
            y=dff.Weekly_Sales,
            name="Actual Weekly Sales",
            line=dict(color="maroon", width=3)
        ),
        secondary_y=False
    )
    
    # Holiday Flag
    fig.add_trace(
        go.Scatter(
            x=dff.Time_Format,
            y=dff.IsHoliday_x,
            name="Holiday Flag",
            line=dict(color="darkturquoise", width=3, dash="dash")
        ),
        secondary_y=True
    )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=predict.ds,
            y=predict.yhat,
            name="Forecast",
            line=dict(color="grey", width=3)
        ),
        secondary_y=False
    )

    # Forecast Lower Bound
    fig.add_trace(
        go.Scatter(
            x=predict.ds,
            y=predict.yhat_lower,
            name="Forecast Lower",
            line=dict(color="red", width=2, dash="dash")
        ),
        secondary_y=False
    )

    # Forecast Upper Bound
    fig.add_trace(
        go.Scatter(
            x=predict.ds,
            y=predict.yhat_upper,
            name="Forecast Upper",
            line=dict(color="blue", width=2, dash="dash")
        ),
        secondary_y=False
    )

    # Update layout of the figure
    return fig.update_layout(
        title="System Load: Actual vs. Holiday Flag",
        xaxis_title="Date",
        yaxis_title="Sales",
        template=TEMPLATE,
    )

# Run the app if the script is executed
if __name__ == '__main__':
    app.run(debug=True)
