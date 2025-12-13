#Reading the datasets
import pandas as pd
file_path = r'D:\DataScience Final Project\Water_Quality_Portability\water_quality_potability.csv'
water_df = pd.read_csv(file_path)
# print(water_df.head())
# print(water_df.tail())
# print(water_df.describe())

#Cleaning the datasets
# print(water_df.dtypes)


for column in water_df.columns:
    unique_values = water_df[column].unique()
    # print(f"Column: {column}")
    # print(f"Unique values: {unique_values}")
    
#Incase duplicate values and missing values arises 
water_df = water_df.drop_duplicates()
water_df = water_df.dropna()

#incase outliers arises (i use boxplot)
import matplotlib.pyplot as plt
import numpy as np

# fig, ax = plt.subplots(figsize=(10, 6))
# water_df.boxplot(ax=ax)
# plt.title('Box Plot for Outlier Detection')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
#from plot i can find outlier in the Solids column
Q1 = water_df.Solids.quantile(0.25)
Q3 = water_df.Solids.quantile(0.75)
print(Q1,Q3)
IQR = Q3-Q1
lower_limit = Q1-1.5*IQR
upper_limit = Q3+1.5*IQR
#detection
# water_df = water_df[(water_df.Solids<lower_limit)|(water_df.Solids>upper_limit)]
# print(water_df)
#removing outliers
water_df = water_df[(water_df.Solids>lower_limit)&(water_df.Solids<upper_limit)]
print(water_df)
#After the removal of outliers now i have 8377 rows of data it was about 10 thousand data at beginning
#Now my data is totally clean 

#Visualization of the data 
import streamlit as st

# Set page config for wider displaying
st.set_page_config(page_title="Water Data Analysis by Arbind", layout="wide")

# Display the data
st.title("Water Quality Analysis Dashboard")
st.write(f"Dataset shape: {water_df.shape}")
st.dataframe(water_df)






st.header("Interactive Plots")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Interactive Barchart")
    st.subheader('Choose Your Parameter')
    
    parameter1 = st.selectbox(
        'Select water quality parameter:',
        ['ph', 'Hardness', 'Solids', 'Turbidity', 'Trihalomethanes'],
        key='barchart_param'
    )
    
    num_bars = st.slider('Number of bars:', 5, 50, 20, key='num_bars')
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))  
    bars = ax1.bar(range(num_bars), water_df[parameter1].head(num_bars))
    ax1.set_title(f'{parameter1} Values (First {num_bars} samples)')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel(parameter1)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.header("2. Interactive Boxplot")
    st.subheader('Choose Your Parameter')
    
    parameter2 = st.selectbox(
        'Select water quality parameter:',
        ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
         'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
        key='boxplot_param'
    )
    
    fig2, ax2 = plt.subplots(figsize=(5.7, 4)) 
    bp = ax2.boxplot(water_df[parameter2])
    mean_value = water_df[parameter2].mean()
    ax2.axhline(y=mean_value, color='red', linestyle='--', alpha=0.7, 
               label=f'Mean: {mean_value:.2f}')
    ax2.set_ylabel(parameter2)
    ax2.set_title(f'Distribution of {parameter2}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig2)





col1, col2 = st.columns(2)

with col1:
    st.subheader('3.Interactive Scatterplot')
 
    scatter_col1, scatter_col2 = st.columns(2)
    with scatter_col1:
        x_axis = st.selectbox('X-axis:', 
                              ['ph', 'Hardness', 'Solids', 'Chloramines', 
                               'Sulfate', 'Conductivity', 'Organic_carbon', 
                               'Trihalomethanes', 'Turbidity'],
                              key='scatter_x')
    with scatter_col2:
        y_axis = st.selectbox('Y-axis:', 
                              ['ph', 'Hardness', 'Solids', 'Chloramines', 
                               'Sulfate', 'Conductivity', 'Organic_carbon', 
                               'Trihalomethanes', 'Turbidity'],
                              key='scatter_y')
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(water_df[x_axis], water_df[y_axis], 
                alpha=0.6, s=15, color='green')
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel(y_axis)
    ax1.set_title(f'{x_axis} vs {y_axis}')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('4.Interactive Histogram')
    
    parameter = st.selectbox(
        'Select water quality parameter:',
        ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
         'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
        key='hist_param'
    )
    
    num_bins = st.slider('Number of bins:', 5, 50, 20, key='hist_bins')
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    n, bins, patches = ax2.hist(water_df[parameter], 
                               bins=num_bins, 
                               edgecolor='black', 
                               alpha=0.7,
                               color='aqua')
    
    ax2.set_xlabel(parameter)
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of {parameter}')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)









st.header("Other Sample Plots")
st.header("Linecharts")


col1, col2 = st.columns(2)
with col1:
    st.write('1. pH Level Analysis')
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(water_df.index, water_df['ph'], label='pH Level', color='blue', linewidth=2)
    ax1.axhline(y=7.0, color='red', linestyle='--', alpha=0.5, label='Neutral (pH 7.0)')
    ax1.axhline(y=6.5, color='orange', linestyle='--', alpha=0.3, label='Lower Limit')
    ax1.axhline(y=8.5, color='orange', linestyle='--', alpha=0.3, label='Upper Limit')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('pH Level')
    ax1.set_title('pH Variation', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.write('2. pH vs Turbidity (First 50)')
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(water_df.index[:50], water_df['ph'].head(50), label='pH', color='blue', linewidth=1.5)
    ax2.plot(water_df.index[:50], water_df['Turbidity'].head(50), label='Turbidity', color='red', linewidth=1.5)
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Measurement')
    ax2.set_title('pH vs Turbidity', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    

col3, col4 = st.columns(2)

with col3:
    st.subheader('Raw vs Moving Average (pH)')
    window = st.slider('Smoothing window size:', 1, 20, 5, key='window_slider')
    
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(water_df.index[:100], water_df['ph'].head(100), 
             label='Raw pH', alpha=0.6, color='blue', linewidth=1)
    ax3.plot(water_df.index[:100], water_df['ph'].head(100).rolling(window).mean(), 
             label=f'Moving Avg (window={window})', linewidth=2.5, color='red')
    ax3.set_xlabel('Sample Number')
    ax3.set_ylabel('pH Level')
    ax3.set_title('pH: Raw vs Smoothed')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

with col4:
    st.subheader('Multiple Parameters Comparison')
    fig4, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0,0].plot(water_df.index[:100], water_df['ph'].head(100), color='blue')
    axes[0,0].set_title('pH Level', fontsize=10)
    axes[0,0].grid(True, alpha=0.2)
    
    axes[0,1].plot(water_df.index[:100], water_df['Turbidity'].head(100), color='orange')
    axes[0,1].set_title('Turbidity', fontsize=10)
    axes[0,1].grid(True, alpha=0.2)
    
    axes[1,0].plot(water_df.index[:100], water_df['Hardness'].head(100), color='green')
    axes[1,0].set_title('Hardness', fontsize=10)
    axes[1,0].grid(True, alpha=0.2)
    
    axes[1,1].plot(water_df.index[:100], water_df['Trihalomethanes'].head(100), color='purple')
    axes[1,1].set_title('Trihalomethanes', fontsize=10)
    axes[1,1].grid(True, alpha=0.2)
    
    plt.suptitle('Four Key Water Parameters', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig4)
    
    
    
    
    
    
#Barcharts
st.header("Barcharts")
col1, col2 = st.columns(2)

with col1:
    st.subheader('1. Average Values of Parameters')
    avg_values = {
        'pH': water_df['ph'].mean(),
        'Turbidity': water_df['Turbidity'].mean(),
        'Hardness': water_df['Hardness'].mean(),
        'Trihalomethanes': water_df['Trihalomethanes'].mean()
    }
    
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    bars = ax1.bar(avg_values.keys(), avg_values.values(), 
                   color=['blue', 'orange', 'green', 'purple'])
    ax1.set_ylabel('Average Value')
    ax1.set_title('Average Water Quality Parameters')
    ax1.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('2. Average Values by Potability')
    potable_avg = water_df[water_df['Potability'] == 1][['ph', 'Turbidity']].mean()
    non_potable_avg = water_df[water_df['Potability'] == 0][['ph', 'Turbidity']].mean()
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, potable_avg.values, width, 
                   label='Potable (1)', color='aqua', edgecolor='black')
    bars2 = ax2.bar(x + width/2, non_potable_avg.values, width, 
                   label='Non-Potable (0)', color='red', edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['pH', 'Turbidity'])
    ax2.set_ylabel('Average Value')
    ax2.set_title('Comparison by Potability')
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    
    

    
    
    
    
    
    
    
    
#Boxplots
st.header("Boxplots")
col1, col2 = st.columns(2)
with col1:
    st.subheader('1.pH Levels Distribution')
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.boxplot(water_df['ph'])
    ax1.set_ylabel('pH Value')
    ax1.set_title('Overall pH Distribution')
    ax1.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig1)

with col2:
    st.subheader('2.pH by Potability')
    potable_ph = water_df[water_df['Potability'] == 1]['ph']
    non_potable_ph = water_df[water_df['Potability'] == 0]['ph']
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    box = ax2.boxplot([potable_ph, non_potable_ph], 
                     tick_labels=['Potable', 'Non-Potable'],
                     patch_artist=True)
    ax2.set_ylabel('pH Value')
    ax2.set_title('pH by Potability Status')
    ax2.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig2)
    
    

    
    
    
    
    
    
#Piecharts
st.header("Piecharts")
col1, col2 = st.columns(2)

with col1:
    st.subheader('1. Potability Distribution')
    potable_count = water_df['Potability'].sum()
    non_potable_count = len(water_df) - potable_count
    
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax1.pie(
        [potable_count, non_potable_count], 
        labels=['Potable (1)', 'Non-Potable (0)'],
        autopct='%1.1f%%',
        colors=['green', 'darkred'],
    )
    ax1.set_title('Water Potability Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('2. pH Level Categories')
    
    def categorize_ph(pH):
        if pH < 6.5:
            return 'Acidic (<6.5)'
        elif 6.5 <= pH <= 8.5:
            return 'Neutral (6.5-8.5)'
        else:
            return 'Alkaline (>8.5)'
    
    water_df['pH_category'] = water_df['ph'].apply(categorize_ph)
    category_counts = water_df['pH_category'].value_counts()
    
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(category_counts.values, 
           labels=category_counts.index,
           autopct='%1.1f%%',
           colors=['aqua', 'lightgreen', 'red'],
          )
    ax2.set_title('pH Level Categories', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    
    
    
    
    
#Scatterplots
st.header("Scatter plots")
col1, col2 = st.columns(2)

with col1:
    st.subheader('pH vs Turbidity')
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(water_df['ph'], water_df['Turbidity'], 
               alpha=0.6, color='blue', s=20)
    ax1.set_xlabel('pH Level')
    ax1.set_ylabel('Turbidity')
    ax1.set_title('pH vs Turbidity')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader('pH vs Hardness by Potability')
    
    potable = water_df[water_df['Potability'] == 1]
    non_potable = water_df[water_df['Potability'] == 0]
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.scatter(potable['ph'], potable['Hardness'], 
               color='green', alpha=0.6, label='Potable', s=20)
    ax2.scatter(non_potable['ph'], non_potable['Hardness'], 
               color='red', alpha=0.6, label='Non-Potable', s=20)
    
    ax2.set_xlabel('pH Level')
    ax2.set_ylabel('Hardness')
    ax2.set_title('pH vs Hardness')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    



#4. Train the different ML model for you problem and show barchart for their accuracy metrics [KNN, SVC, Logistic Regrssion] (best model after hyperparameter tuning) (In Streamlit Dashboard)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

X = water_df[feature_names]
y = water_df['Potability']
print(feature_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,stratify=y)

#A. Knn model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
#tuning hyper-parameter by grid search cross validation because we have smaller dataset
# model = KNeighborsClassifier()
# parameter_grid = {
#     "n_neighbors" : range(2,21),
#     "weights": ["uniform","distance"],
#     "metric":["euclidean","manhattan"]
# }
# grid_search = GridSearchCV(model, parameter_grid, cv=5, scoring="recall")
# grid_search.fit(X_scaled,y)
# print(grid_search.best_score_)
# print(grid_search.best_params_)

#0.8677313664224011 {'metric': 'manhattan', 'n_neighbors': 17, 'weights': 'distance'} so our hyperparameter should be 17

# model= KNeighborsClassifier(n_neighbors=17)
# model.fit(X_train,y_train)
# y_pred = model.predict(X_test)

# recall_score_value = recall_score(y_test, y_pred)
# print(f"Recall Score is :{recall_score_value} ")
#Recall Score is :0.8581314878892734 


#B. Svm model
from sklearn.svm import SVC 

model = SVC(random_state=42)
# parameter_grid = {
#     "C": [0.1, 1, 10, 100],  # Regularization parameter
#     "kernel": ["linear", "rbf", "poly", "sigmoid"],  # Kernel type
#     "gamma": ["scale", "auto", 0.1, 0.01, 0.001],  # Kernel coefficient
#     "degree": [2, 3, 4, 5],  # Degree for poly kernel (only used when kernel='poly')
# } this standard parameters took a lot of time to display the output so i eradicated its value

parameter_grid = {
    "C": [0.1, 1, 10],  # Reduced from 4 to 3
    "kernel": ["linear", "rbf"],  # Removed poly, sigmoid (often less effective)
    "gamma": ["scale", "auto"],  # Removed numeric values
    # Removed degree entirely
}
# grid_search = GridSearchCV(model, parameter_grid, cv=5, scoring="recall")
# grid_search.fit(X_scaled,y)
# print(grid_search.best_score_)
# print(grid_search.best_params_)
#0.8716540005487319 {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}

model = SVC(C=1,gamma='scale',kernel='rbf',random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

recall_score_value = recall_score(y_test, y_pred)
print(f"Recall Score is :{recall_score_value} ")
#Recall Score is :0.8719723183391004 

#On comparing i found SVM model is better for my relevant data 
#now on streamlit dashboard 
# water_type_mapping = {
#     0 : "Not-Portable (Unsafe to drink)",
#     1 : "Portable (Safe to drink)"
# }
# user_input = input("Enter 'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity' values separated by comma :")
# user_data_list = user_input.split(",")
# user_data = map(float, user_data_list)

# user_data = [list(map(float,user_data_list))]
# new_data = np.array(user_data)

# predicted_label = int(model.predict(new_data))
# print(f"Your water type for given data is : {water_type_mapping.get(predicted_label)}")




water_type_mapping = {
    0: "Not-Potable (Unsafe to drink)",
    1: "Potable (Safe to drink)"
}
st.write("Enter water quality parameters:")

#input fields
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    hardness = st.number_input("Hardness", min_value=0.0, value=195.0, step=0.1)
    solids = st.number_input("Solids", min_value=0.0, value=22000.0, step=10.0)
    chloramines = st.number_input("Chloramines", min_value=0.0, value=7.1, step=0.1)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=330.0, step=1.0)

with col2:
    conductivity = st.number_input("Conductivity", min_value=0.0, value=420.0, step=1.0)
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=14.3, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=66.0, step=0.1)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=3.9, step=0.1)
    
# Prediction button
if st.button("🔍 Predict Water Potability", type="primary"):
    # Prepare input data
    user_data = [[ph, hardness, solids, chloramines, sulfate, 
                  conductivity, organic_carbon, trihalomethanes, turbidity]]
    
    # Scale the input data (important for SVM)
    user_data_scaled = scaler.transform(user_data)
    
    # Make prediction
    predicted_label = int(model.predict(user_data_scaled)[0])
    
    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    if predicted_label == 1:
        st.success(water_type_mapping[predicted_label])
    else:
        st.error(water_type_mapping[predicted_label])
    