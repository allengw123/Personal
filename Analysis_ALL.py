# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:13:00 2024

@author: allen
"""

#%% Functions

import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.io as pio
import numpy as np

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


def plot_regression_with_stats(input_data, x_column, y_column, name_column):
    
    data = input_data[[x_column,y_column] + name_column].dropna()
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
        xaxis_title=extract_titles(x_column)[-1],
        yaxis_title=extract_titles(y_column)[-1],
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
    
# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # Radius of the Earth in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
#%% 

# Specify variables
hospital_path = 'C:/Users/allen/Documents/Abbott/Implant Facility.csv'
communityneuro_path = 'C:/Users/allen/Documents/Abbott/Community Neurologist.csv'

# Load data
hospital_data = load_csv(hospital_path)
communityneuro_data = load_csv(communityneuro_path)

miles = 50
hospital_comneuro_practice = []
hospital_comneuro_count = []
for idx, hos in hospital_data.iterrows():
    hospital_longitude = hos["Practice Location Longitude"]
    hospital_latitude = hos["Practice Location Latitude"]
    local_neuro = []
    local_neuro_count = 0
    for idx, communityneuro in communityneuro_data.iterrows():
        communityneuro_longitude = communityneuro["Practice Location Longitude"]
        communityneuro_latitude = communityneuro["Practice Location Latitude"]
        distance = haversine(hospital_latitude, hospital_longitude,
                             communityneuro_latitude, communityneuro_longitude)
        if distance < miles:
            local_neuro.append(communityneuro["Affiliated Practice"])
            local_neuro_count += 1
    hospital_comneuro_practice.append(local_neuro)
    hospital_comneuro_count.append(local_neuro_count)

hospital_data["Local Community Neurologist Practices - Affiliated Practice"] = hospital_comneuro_practice
hospital_data["Local Community Neurologist Practices - Count"] = hospital_comneuro_count           
            

# Create regression
plot_regression_with_stats(hospital_data,
                           "Local Community Neurologist Practices - Count",
                           "Care Clusters - Parkinson's Disease - Total Patients",
                           ['Organization Legal Name'])
    
#