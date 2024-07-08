# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:47:36 2024

@author: allen
"""
#%% Functions

import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'


def extract_titles(input_str):
    return input_str.split(' - ')
    
def get_column_data(data, substring):
    columns_with_substring = data.filter(like=substring)
    return columns_with_substring


def filter_data_by_substring(data, column_name, substring):
    return data[data[column_name].str.contains(substring, na=False)]

def load_csv(file_path):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        print("No data: The file is empty")
    except pd.errors.ParserError:
        print("Parse error: Check the file format")
    except Exception as e:
        print(f"An error occurred: {e}")
    return data


def plot_regression_with_stats(data, x_column, y_column, name_column):
    
    data = data[[x_column,y_column] + name_column].dropna()
    data[x_column] = pd.to_numeric(data[x_column], errors='coerce')
    data[y_column] = pd.to_numeric(data[y_column], errors='coerce')
    data = data.dropna()

    # Fit the regression model
    X = data[[x_column]]
    y = data[y_column]
    
    model = sm.OLS(y, X).fit()

    # Get R-squared and p-value
    r_squared = model.rsquared
    p_value = model.pvalues[0]  # p-value for the x_column coefficient

    # Calculate predicted values
    data['predicted'] = model.predict(X)

    # Add a new column for label text
    if len(name_column)<2:
        data['label_text'] = data.apply(lambda row: row[name_column] if row[y_column] < row['predicted'] else '', axis=1)


    # Define colors for each category
    data['Category'] = data.apply(lambda row: 'Below Regression' if row[y_column] < row['predicted'] else 'Above Regression', axis=1)
    color_discrete_map = {
        'Below Regression': 'red',  # Red for below regression line
        'Above Regression': 'blue'  # Blue for above regression line
    }


    # Plot the data
    h_data = {header: header in name_column for header in data.columns}
    
    if len(name_column)<2:
        fig = px.scatter(
            data, 
            x = x_column,
            y = y_column,
            color = 'Category',
            text = 'label_text',
            color_discrete_map = color_discrete_map,
            hover_data = h_data
        )
    else:
        fig = px.scatter(
            data, 
            x = x_column,
            y = y_column,
            color = 'Category',
            color_discrete_map = color_discrete_map,
            hover_data = h_data
        )
    # Add the regression line
    regression_line = pd.DataFrame({x_column: data[x_column], y_column: model.predict(X)})
    fig.add_traces(
        px.line(
            regression_line, x=x_column, y=y_column
        ).data
    )

    # Adjust layout to prevent labels from going off-screen
    fig.update_traces(
        textposition='bottom center',
        cliponaxis=False  # Prevents text labels from being clipped
    )

    
    # Add titles and labels
    fig.update_layout(
        title=extract_titles(x_column)[1],
        xaxis_title=extract_titles(x_column)[2],
        yaxis_title=extract_titles(y_column)[2],
        annotations=[dict(
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            text=f'R-squared = {r_squared:.2f}<br>p-value = {p_value:.2e}',
            showarrow=False,
            font=dict(size=12)
        )]
    )
    # Show the plot
    fig.show()  
    
#%%

# Specify the file path
file_path = 'C:/Users/allen/Documents/Abbott/Implant Facility.csv'

# Call the function to load the CSV
data = load_csv(file_path)

# Filter by state
data = filter_data_by_substring(data, 'State (Practice)', 'OH')

# Get data
column_headers = data.columns
total_patients = get_column_data(data, 'Total Patients')
total_revenue = get_column_data(data, 'Revenue')

# Create regression
for x,y in zip(total_patients.columns,total_revenue.columns):
    plot_regression_with_stats(data,x,y,['Organization Legal Name'])
    
#%%
# Specify the file path
file_path = 'C:/Users/allen/Documents/Abbott/Community Neurologist.csv'

# Call the function to load the CSV
data = load_csv(file_path)

# Filter by state
data = filter_data_by_substring(data, 'State (Practice)', 'OH')

# Get data
column_headers = data.columns
total_patients = get_column_data(data, 'Total Patients')
total_revenue = get_column_data(data, 'Revenue')

# Create regression
for x,y in zip(total_patients.columns,total_revenue.columns):
    plot_regression_with_stats(data,x,y,['First name','Last name','Affiliated Practice'])