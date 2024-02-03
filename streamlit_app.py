import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title('Regresión lineal')
st.markdown('*Autor: Gabriel Barragán*')

st.write('# Cargar base de datos')

# Load Data
# Create a list of datasets
datasets = ["1.longitud_femur_estatura.csv",
           "6.extension_hielo.csv"
           ]

# Create a dropdown menu to select the dataset
selected_dataset = st.selectbox("Seleccione una base de datos", datasets)

# Read the selected dataset into a pandas Dataframe
df = pd.read_csv('Datasets/'+selected_dataset)

# Display the Dataframe
if st.checkbox('Mostrar base de datos'):
           st.write('Base de datos: '+selected_dataset)
           st.dataframe(df)

# Access X and y variables
X = df.iloc[:,0]
y = df.iloc[:,1]

# Plot a scatterplot
if st.checkbox('Mostrar diagrama de dispersión'):
           st.write('# Diagrama de dispersión')
           plt.subplots()
           plt.title('Diagrama de dispersión')
           plt.scatter(X,y)
           plt.xlabel(X.name)
           plt.ylabel(y.name)
           # Display the plot in Streamlit
           st.pyplot(plt)


# Create a linear regression model
if st.checkbox('Calcular regresión lineal'):
           st.write('# Diagrama de dispersión')
           model = LinearRegression()

           # Fit the model to the data
           model.fit(X.values.reshape(-1,1), y)

           intercept = model.intercept_
           coefficient = model.coef_[0]
           st.write(f'Modelo de regressión lineal: y = {coefficient:.4f}x + {intercept:.4f}')

           # Plot the regression line
           x_vals = X.values.reshape(-1,1)
           y_vals = model.predict(x_vals)
           plt.subplots()
           plt.title('Diagrama de dispersión y recta de regresión')
           plt.scatter(X, y)
           plt.plot(x_vals, y_vals, color='red')
           plt.xlabel(X.name)
           plt.ylabel(y.name)
           # Display the plot in Streamlit
           st.pyplot(plt)

           # Predict a new value
           st.write('# Predicción de valores con el modelo de regresión lineal')
           st.write('X: ',X.name)
           input_value = st.number_input('Introduce un valor de X', 10)

           predicted_value = model.predict([[input_value]])
           st.write(f'Si {X.name} es {input_value} entonces {y.name} es {predicted_value[0]:.2f}')
