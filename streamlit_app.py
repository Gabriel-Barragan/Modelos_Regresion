import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.title('Regresión lineal')
st.markdown('*Autor: Gabriel Barragán*')

st.write('# Cargar base de datos')

# Load Data
# Create a list of datasets
datasets = ["1.longitud_femur_estatura.csv",
            "2.Demanda_bebidas_gaseosas.csv",
            "3.Diametro_arbol.csv",
            "4.Nivel_CO2.csv",
            "5.Temperatura_grillos.csv",
            "6.extension_hielo.csv",
            "7.Prevalencia_mosquitos.csv",
            "8.Ruido_inteligibilidad.csv",
            "9.Esperanza_vida.csv",
            "11.Record_Olimpico.csv"
           ]

# Create a dropdown menu to select the dataset
selected_dataset = st.selectbox("Seleccione una base de datos", datasets)

# Read the selected dataset into a pandas Dataframe
df = pd.read_csv('Datasets/'+selected_dataset)

if selected_dataset != '11.Record_Olimpico.csv':
            st.write('# Datos no agrupados')

            # Display the Dataframe
            if st.checkbox('Mostrar base de datos'):
                       st.write('Base de datos: '+selected_dataset)
                       st.dataframe(df)

            if st.checkbox('Mostrar estadísticos descriptivos'):
                        st.write(df.describe())
            
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
                       st.write('# Modelo de regresión lineal')
                       model = LinearRegression()

                       # Fit the model to the data
                       model.fit(X.values.reshape(-1,1), y)

                       intercept = model.intercept_
                       coefficient = model.coef_[0]
                       st.write('x: ',X.name)
                       st.write('y: ',y.name)
                       st.write(f'Modelo de regresión lineal: y = {coefficient:.4f}x + {intercept:.4f}')
                       correlation_coef = df[X.name].corr(df[y.name])
                       st.write(f'Coeficiente de correlación: r = {correlation_coef:.2f}')
           
                       # Plot the regression line
                       #x_vals = X.values.reshape(-1,1)
                       #y_vals = model.predict(x_vals)
                       #plt.subplots()
                       #plt.title('Diagrama de dispersión y recta de regresión')
                       #plt.scatter(X, y)
                       #plt.plot(x_vals, y_vals, color='red')
                       #plt.xlabel(X.name)
                       #plt.ylabel(y.name)
                       # Display the plot in Streamlit
                       #st.pyplot(plt)
            
                       x_min = st.number_input('Valor mínimo x:',value=X.min())
                       x_max = st.number_input('Valor máximo x:',value=X.max())
                       x_range_prediction = np.arange((0.99)*x_min,(1.01)*x_max,1)
                       y_range_prediction = model.predict(x_range_prediction.reshape(-1,1))
                       plt.subplots()
                       plt.title('Diagrama de dispersión y recta de regresión')
                       plt.scatter(X, y)
                       plt.plot(x_range_prediction, y_range_prediction, color='red')
                       plt.xlabel(X.name)
                       plt.ylabel(y.name)
                       # Display the plot in Streamlit
                       st.pyplot(plt)

                       # Predict a new value
                       st.write('# Predicción de valores con el modelo de regresión lineal')
                       st.write('x: ',X.name)
                       input_value = st.number_input('Introduce un valor de x', value=X.min())

                       predicted_value = model.predict([[input_value]])
                       st.write(f'Si {X.name} es {input_value} entonces {y.name} es {predicted_value[0]:.2f}')
else: 
            st.write('# Datos agrupados')
            # Display the Dataframe
            if st.checkbox('Mostrar base de datos'):
                       st.write('Base de datos: '+selected_dataset)
                       st.dataframe(df)
            
            if st.checkbox('Mostrar estadísticos descriptivos'):
                        st.write(df.groupby('Sexo')['Tiempo'].describe())

                        # Bar chart of number of athletes per country, grouped by sex
                        plt.subplots()
                        plt.title('Número de atletas por país agrupados por sexo')
                        sns.histplot(data=df, y="Pais", hue="Sexo", stat="count", multiple="dodge", shrink=0.75)
                        st.pyplot(plt)

            
            if st.checkbox('Mostrar diagrama de dispersión'):
                        st.write('# Diagrama de dispersión')
                        plt.subplots()
                        plt.title('Diagrama de dispersión')
                        sns.scatterplot(data=df, x="Anio", y="Tiempo", hue="Sexo")
                        st.pyplot(plt)

            # Create a linear regression model
            if st.checkbox('Calcular regresión lineal'):
                        st.write('# Modelo de regresión lineal')
                        grouped_data = df.groupby('Sexo')
                        models = {}

                        plt.subplots()
                        for name, group in grouped_data:
                                    model = LinearRegression()
                                    model.fit(group['Anio'].values.reshape(-1,1), group['Tiempo'])
                                    models[name] = model
                                    plt.scatter(group['Anio'], group['Tiempo'], label=name)
                                    x_vals = np.linspace(group['Anio'].min(), group['Anio'].max())
                                    y_vals = models[name].predict(x_vals.reshape(-1,1))
                                    plt.plot(x_vals, y_vals, label=f"Recta regresión: {name}")

                        st.pyplot(plt)
