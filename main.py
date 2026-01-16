#Reading the datasets
import pandas as pd
file_path = r'water_quality_potability.csv'
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

# ============================================================================
# 4. Train the different ML model for you problem and show barchart for their 
#    accuracy metrics [KNN, SVC, Logistic Regression] 
#    (best model after hyperparameter tuning) (In Streamlit Dashboard)
# ============================================================================
st.header("ML Model Comparison Results")

# Import model comparison results from test.py
try:
    from test import compare_models, train_single_model
    
    # Initialize session state variables
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'metrics_data' not in st.session_state:
        st.session_state.metrics_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'default_model_trained' not in st.session_state:
        st.session_state.default_model_trained = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = 'SVM (Default)'
    
    # Prepare features and target for default training
    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    st.session_state.feature_names = feature_names
    
    # Train default SVM model on app startup (only once)
    if not st.session_state.default_model_trained:
        with st.spinner("Training default SVM model..."):
            # Train SVM as default model
            X = water_df[feature_names]
            y = water_df['Potability']
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, stratify=y, random_state=42
            )
            
            # Train SVM model with best parameters
            from sklearn.svm import SVC
            model = SVC(C=1, gamma='scale', kernel='rbf', random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions and calculate metrics
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store in session state
            st.session_state.metrics_data = {
                'SVM': {
                    'model': model,
                    'accuracy': round(accuracy * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'f1': round(f1 * 100, 2),
                    'best_params': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
                }
            }
            st.session_state.scaler = scaler
            st.session_state.best_model = model
            st.session_state.model_trained = True
            st.session_state.default_model_trained = True
            st.session_state.selected_model = 'SVM (Default)'
    
    # Create two tabs for training and prediction
    tab1, tab2 = st.tabs(["Model Training", "Real-time Prediction"])
    
    with tab1:
        st.subheader("Train Machine Learning Models")
        
        # Show default model status
        if st.session_state.default_model_trained:
            st.success("✓ Default SVM model trained automatically on startup")
        
        # Train all models button
        if st.button("Train All Models", key="train_all_models"):
            with st.spinner("Training all models..."):
                results_df, scaler, models = compare_models()
                
                # Store in session state
                st.session_state.metrics_data = {
                    'KNN': {
                        'model': models['KNN'],
                        'accuracy': results_df[results_df['Model'] == 'KNN']['Accuracy'].values[0],
                        'precision': results_df[results_df['Model'] == 'KNN']['Precision'].values[0],
                        'recall': results_df[results_df['Model'] == 'KNN']['Recall'].values[0],
                        'f1': results_df[results_df['Model'] == 'KNN']['F1-Score'].values[0],
                        'best_params': {'n_neighbors': 17, 'metric': 'manhattan', 'weights': 'distance'}
                    },
                    'SVM': {
                        'model': models['SVM'],
                        'accuracy': results_df[results_df['Model'] == 'SVM']['Accuracy'].values[0],
                        'precision': results_df[results_df['Model'] == 'SVM']['Precision'].values[0],
                        'recall': results_df[results_df['Model'] == 'SVM']['Recall'].values[0],
                        'f1': results_df[results_df['Model'] == 'SVM']['F1-Score'].values[0],
                        'best_params': {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
                    },
                    'Logistic Regression': {
                        'model': models['Logistic Regression'],
                        'accuracy': results_df[results_df['Model'] == 'Logistic Regression']['Accuracy'].values[0],
                        'precision': results_df[results_df['Model'] == 'Logistic Regression']['Precision'].values[0],
                        'recall': results_df[results_df['Model'] == 'Logistic Regression']['Recall'].values[0],
                        'f1': results_df[results_df['Model'] == 'Logistic Regression']['F1-Score'].values[0],
                        'best_params': {'max_iter': 1000}
                    }
                }
                st.session_state.scaler = scaler
                st.session_state.model_trained = True
                
                # Find and store the best model
                best_model_row = results_df.loc[results_df['Recall'].idxmax()]
                best_model_name = best_model_row['Model']
                st.session_state.best_model = models[best_model_name]
                st.session_state.selected_model = best_model_name
                
                st.success(f"All models trained successfully! Best model: {best_model_name}")
        
        # Train individual model buttons
        st.subheader("Train Individual Models")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Train KNN", key="train_knn"):
                with st.spinner("Training KNN..."):
                    # Call function to train only KNN
                    model, scaler, metrics = train_single_model('KNN')
                    if model:
                        st.session_state.metrics_data['KNN'] = metrics['KNN']
                        st.session_state.scaler = scaler
                        st.session_state.model_trained = True
                        st.session_state.best_model = model
                        st.session_state.selected_model = 'KNN'
                        st.success("KNN trained successfully!")
        
        with col2:
            if st.button("Train SVM", key="train_svm"):
                with st.spinner("Training SVM..."):
                    # Call function to train only SVM
                    model, scaler, metrics = train_single_model('SVM')
                    if model:
                        st.session_state.metrics_data['SVM'] = metrics['SVM']
                        st.session_state.scaler = scaler
                        st.session_state.model_trained = True
                        st.session_state.best_model = model
                        st.session_state.selected_model = 'SVM'
                        st.success("SVM trained successfully!")
        
        with col3:
            if st.button("Train Logistic Regression", key="train_lr"):
                with st.spinner("Training Logistic Regression..."):
                    # Call function to train only Logistic Regression
                    model, scaler, metrics = train_single_model('Logistic Regression')
                    if model:
                        st.session_state.metrics_data['Logistic Regression'] = metrics['Logistic Regression']
                        st.session_state.scaler = scaler
                        st.session_state.model_trained = True
                        st.session_state.best_model = model
                        st.session_state.selected_model = 'Logistic Regression'
                        st.success("Logistic Regression trained successfully!")
        
        # Display training results if models are trained
        if st.session_state.model_trained and st.session_state.metrics_data:
            st.subheader("Model Performance Metrics")
            
            # Create results DataFrame for display
            results_list = []
            for model_name, data in st.session_state.metrics_data.items():
                results_list.append({
                    'Model': model_name,
                    'Accuracy': data['accuracy'],
                    'Precision': data['precision'],
                    'Recall': data['recall'],
                    'F1-Score': data['f1']
                })
            
            results_df = pd.DataFrame(results_list)
            
            # Display results table
            st.dataframe(results_df)
            
            # Visualization of model comparison
            st.subheader("Model Comparison Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data for bar chart
            model_names = results_df['Model'].tolist()
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            x = np.arange(len(model_names))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                values = results_df[metric].tolist()
                ax.bar(x + i*width, values, width, label=metric)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score (%)')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width*1.5)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display best parameters
            st.subheader("Best Hyperparameters Found:")
            for model_name, data in st.session_state.metrics_data.items():
                st.write(f"**{model_name}:** {data['best_params']}")
            
            # Find and display best model
            if len(st.session_state.metrics_data) > 0:
                best_model_name = max(st.session_state.metrics_data.keys(), 
                                    key=lambda x: st.session_state.metrics_data[x]['recall'])
                st.subheader(f"Best Model: {best_model_name}")
                st.write(f"**Recall Score:** {st.session_state.metrics_data[best_model_name]['recall']:.2f}%")
        else:
            if st.session_state.default_model_trained:
                st.info("Default SVM model is trained. You can train other models using the buttons above.")
            else:
                st.info("No models trained yet. Click the buttons above to train models.")
    
    with tab2:
        st.subheader("Real-time Water Potability Prediction")
        
        if not st.session_state.model_trained or st.session_state.best_model is None:
            st.warning("No model trained yet. Please train at least one model in the 'Model Training' tab first.")
        else:
            water_type_mapping = {
                0: "Not-Potable (Unsafe to drink)",
                1: "Potable (Safe to drink)"
            }
            
            # Model selection for prediction
            st.subheader("Select Model for Prediction")
            
            # Get available trained models
            available_models = []
            if st.session_state.metrics_data:
                available_models = list(st.session_state.metrics_data.keys())
            
            if available_models:
                # Add default label to SVM if it's the default
                model_options = []
                for model in available_models:
                    if model == 'SVM' and st.session_state.default_model_trained and st.session_state.selected_model == 'SVM (Default)':
                        model_options.append('SVM (Default)')
                    else:
                        model_options.append(model)
                
                selected_model_name = st.selectbox(
                    "Choose model for prediction:",
                    model_options,
                    index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0
                )
                
                # Update selected model
                if selected_model_name != st.session_state.selected_model:
                    # Extract actual model name without "(Default)" suffix
                    actual_model_name = selected_model_name.replace(' (Default)', '')
                    if actual_model_name in st.session_state.metrics_data:
                        st.session_state.best_model = st.session_state.metrics_data[actual_model_name]['model']
                        st.session_state.selected_model = selected_model_name
                        st.rerun()
            
            st.write("Enter water quality parameters for prediction:")
            
            # Input fields
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
            if st.button(" Predict Water Potability", type="primary"):
                # Prepare input data
                user_data = [[ph, hardness, solids, chloramines, sulfate, 
                              conductivity, organic_carbon, trihalomethanes, turbidity]]
                
                # Scale the input data
                user_data_scaled = st.session_state.scaler.transform(user_data)
                
                # Make prediction using selected model
                predicted_label = int(st.session_state.best_model.predict(user_data_scaled)[0])
                
                # Display result
                st.markdown("---")
                st.subheader("Prediction Result")
                
                # Display which model is being used
                st.info(f"Using: {st.session_state.selected_model}")
                
                if predicted_label == 1:
                    st.success(water_type_mapping[predicted_label])
                else:
                    st.error(water_type_mapping[predicted_label])
    
except ImportError:
    st.warning("test.py not found. Please ensure test.py is in the same directory.")
except Exception as e:
    st.error(f"Error loading model comparison: {e}")

# ============================================================================
# Keep all your existing visualization code below (unchanged)
# ============================================================================

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