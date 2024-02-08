import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

st.title('Regresión lineal')
st.markdown('*Autor: Gabriel Barragán*')

tab_titles = ['Regresión lineal','Regresión exponencial','Otros']
tabs = st.tabs(tab_titles)

with tabs[0]:
  st.write('# Cargar base de datos')

  # Load Data
  # Create a list of datasets
  datasets_1 = ["1.longitud_femur_estatura",
              "2.Demanda_bebidas_gaseosas",
              "3.Diametro_arbol",
              "4.Nivel_CO2",
              "5.Temperatura_grillos",
              "6.extension_hielo",
              "7.Prevalencia_mosquitos",
              "8.Ruido_inteligibilidad",
              "9.Esperanza_vida",
              "11.Record_Olimpico"
              ]

  # Create a dropdown menu to select the dataset
  selected_dataset_1 = st.selectbox("Seleccione una base de datos", datasets_1)

  # Read the selected dataset into a pandas Dataframe
  df = pd.read_csv('Datasets/'+selected_dataset_1+'.csv')

  if selected_dataset_1 != '11.Record_Olimpico':
    # Access X and y variables
    X = df.iloc[:,0]
    y = df.iloc[:,1]
    st.write('# Datos no agrupados')

    # Display the Dataframe
    if st.checkbox('Mostrar base de datos'):
      st.write('Base de datos: '+selected_dataset_1)
      st.dataframe(df)

    if st.checkbox('Mostrar estadísticos descriptivos'):
      st.write(df.describe())
      correlation_coef = X.corr(y)
      st.write(f'Coeficiente de correlación entre la variable {df.columns[0]} y {df.columns[1]}: R = {correlation_coef:.2f}')

      #if st.checkbox('Visualización de variable de respuesta'):
      #           st.write(f'# Visualización de {y.name}') 
      #           input_bins = st.slider('Ingrese número de bins', 1, 20, 5)
      #           plt.subplots()
      #           plt.title(f'Histograma de {y.name}') 
      #           plt.hist(x=y, bins=input_bins, edgecolor='black') 
      #           plt.xlabel('Valor')
      #           plt.ylabel('Frecuencia')
      #           st.pyplot(plt)

      #          plt.subplots()
      #          plt.title(f'Boxplot de {y.name}')
      #          plt.boxplot(y) 
      #          st.pyplot(plt)
  
      #          plt.subplots()
      #          #plt.title(f'Q-Q plot de  {y.name}')
      #          stats.probplot(y, dist='norm', plot=plt) 
      #          st.pyplot(plt) 
            
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
                                  
      x_min = st.number_input('Valor mínimo x:',value=X.min())
      x_max = st.number_input('Valor máximo x:',value=X.max()) 
      x_range_prediction = np.arange(x_min, x_max,1)
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
    grouped_data = df.groupby('Sexo')
    # Display the Dataframe
    if st.checkbox('Mostrar base de datos'):
      st.write('Base de datos: '+selected_dataset_1)
      st.dataframe(df)
            
    if st.checkbox('Mostrar estadísticos descriptivos'):
      st.write(grouped_data['Tiempo'].describe())
      
      for name, group in grouped_data:
        st.write(f"Coeficiente de correlación ({name}): R = {group['Anio'].corr(group['Tiempo']):.2f}")
                        
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
        models = {}
        plt.subplots()
        for name, group in grouped_data:
          model = LinearRegression()
          model.fit(group['Anio'].values.reshape(-1,1), group['Tiempo'])
          models[name] = model
          st.write(f"Modelo de regresión lineal ({name}): y= {model.coef_[0]:.4f}x + {model.intercept_:.4f}")
          plt.scatter(group['Anio'], group['Tiempo'], label=name)
          x_vals = np.linspace(group['Anio'].min(), (1.025)*group['Anio'].max())
          y_vals = models[name].predict(x_vals.reshape(-1,1))
          plt.plot(x_vals, y_vals, label=f"Recta regresión: {name}")
            
        plt.xlabel("Anio")
        plt.ylabel("Tiempo")
        plt.title("Diagrama de dispersión y recta de regresión por grupos")
        plt.legend()
        st.pyplot(plt)

with tabs[1]:
  # Load Data
  # Create a list of datasets
  datasets_2 = ["Poblacion_Estados_Unidos",
               "crecimiento_logistico",
               "emisiones_escapes_autos",
               "Especie_area",
               "Experimentos_curva_olvido",
               "Gastos_salud",
               "Ley_Beer_Lambert",
               "modelo_exponencial_o_potencia_1",
               "modelo_exponencial_o_potencia_2",
               "modelo_logaritmico",
               "Pelota_caida",
               "Vida_media"]

  st.write('# Cargar base de datos')
  
  # Create a dropdown menu to select the dataset
  selected_dataset_2 = st.selectbox("Seleccione una base de datos", datasets_2)

  # Read the selected dataset into a pandas Dataframe
  df_2 = pd.read_csv('Datasets/'+selected_dataset_2+'.csv')

  # Display the Dataframe
  if st.checkbox('Mostrar base de datos',value=True):
    st.write('Base de datos: '+selected_dataset_2)
    st.dataframe(df_2)
  
  if selected_dataset_2 == 'Experimentos_curva_olvido':
    conversion_factors = {
      'dia': 24,
      'h': 1,
      'min': 1/60
    }
    df_2['Tiempo_hora'] = df_2.apply(lambda row: row['Tiempo']*conversion_factors[row['DHM']], axis=1)
  
  st.write('# Seleccionar variables y mostrar')
  columns = df_2.columns.tolist()
  selected_columns = st.multiselect('Seleccionar variables', columns)
  filtered_data = df_2[selected_columns]
  st.dataframe(filtered_data.head())

  X = filtered_data.iloc[:,0]
  log_X = np.log(X)
  y = filtered_data.iloc[:,1]
  log_y = np.log(y)

  if st.checkbox('Mostrar estadísticos descriptivos',value=True):
    st.write(filtered_data.describe())
    #correlation_coef = X.corr(y)
    #st.write(f'Coeficiente de correlación entre la variable {X.name} y {y.name}: R = {correlation_coef:.2f}')
  
  if st.checkbox('Diagramas de dispersión',value=True):
    st.write('# Diagramas de dispersión')
    plt.subplots(1,3)
    
    plt.title('Diagrama de dispersión {selected_dataset_2}')
    plt.scatter(X,y)
    plt.xlabel(X.name)
    plt.ylabel(y.name)

    plt.title('Gráfica semi-log {selected_dataset_2}')
    plt.scatter(X,log_y)
    plt.xlabel(X.name)
    plt.ylabel('Log '+y.name)

    plt.title('Gráfica log-log {selected_dataset_2}')
    plt.scatter(log_X,log_y)
    plt.xlabel('Log '+X.name)
    plt.ylabel('Log '+y.name)
    
    # Display the plot in Streamlit
    st.pyplot(plt)

  #if st.checkbox('Mostrar modelo de regresión exponencial'):
