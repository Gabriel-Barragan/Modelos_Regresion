import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import scipy.stats as stats
import seaborn as sns

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title('Modelos de regresión')
st.markdown('*Autor: Gabriel Barragán*')

tab_titles = ['Regresión lineal','Regresión exponencial','Regresión logística','Regresión logarítmica']
tabs = st.tabs(tab_titles)

widget_id = (id for id in range(1, 100_00))

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
    if X.name=='Anio':
      X_min = X.min()
      X = X - X_min
    
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
            
      if X.name=='Anio':
        input_value = st.number_input(f'Introduce número de años después o antes de {X_min}', value=X.min())
        predicted_value = model.predict([[input_value]])
        st.write(f'En el año {X_min+input_value}, se tiene que {y.name} es {predicted_value[0]:.2f}')
      else:
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
  if st.checkbox('Mostrar base de datos', key=next(widget_id)):
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
  if selected_columns:
    filtered_data = df_2[selected_columns]
    st.dataframe(filtered_data.head())

    if st.checkbox('Mostrar estadísticos descriptivos', key=next(widget_id)):
      st.write(filtered_data.describe())
      #correlation_coef = X.corr(y)
      #st.write(f'Coeficiente de correlación entre la variable {X.name} y {y.name}: R = {correlation_coef:.2f}')
    
    if 'Anio' in filtered_data.columns.tolist():
      X = filtered_data.iloc[:,0]
      y = filtered_data.iloc[:,1]
      log_y = np.log(y)
      
      X_min = X.min()
      X = X - X_min

      if st.checkbox('Diagramas de dispersión', key=next(widget_id)):
        st.write('# Diagramas de dispersión')
        fig, axes = plt.subplots(2, 1, figsize=(10,7))
        axes[0].set_title(f'Diagrama de dispersión - '+selected_dataset_2)
        axes[0].scatter(X,y)
        axes[0].set_xlabel(X.name)
        axes[0].set_ylabel(y.name)

        axes[1].set_title(f'Gráfica semi-log - '+selected_dataset_2)
        axes[1].scatter(X,log_y)
        axes[1].set_xlabel(X.name)
        axes[1].set_ylabel('Log '+y.name)

        fig.tight_layout()
    
        # Display the plot in Streamlit
        st.pyplot(fig)

      if st.checkbox('Mostrar modelo de regresión exponencial', key=next(widget_id)):
        model_exponential = LinearRegression()
        model_exponential.fit(X.values.reshape(-1,1), log_y)

        log_C = model_exponential.intercept_
        C = np.exp(log_C)
        k = model_exponential.coef_[0]
    
        st.latex(r'''y = Ce^{kx}  ''')
        st.write('Linearización')
        st.latex(r'''\ln(y) = kx + \ln(C) \quad \Rightarrow \quad Y_{\text{exp}} = kx + A_{\text{exp}}''')
        st.write('donde')
        st.latex(r'''Y_{\text{exp}}=\ln(y),\quad \text{y}\quad A_{\text{exp}}=\ln(C)''')
        st.write('Parámetros:')
        st.latex(r'''k='''+ rf'''{k:.4f}''')
        st.latex(r'''A_{\text{exp}}=\ln(C)=''' + rf'''{log_C:.4f}''' + r'''\quad \Rightarrow \quad C=''' + rf'''{C:.4f}''')

        st.write(f"Modelo de regresión exponencial: $$y = {C:.4f}x^{{{k:.4f}}}$$")

        log_y_predict = model_exponential.predict(X.values.reshape(-1,1))
        R2 = r2_score(log_y,log_y_predict)
        st.write(f'Coeficiente de determinación: $$R^2={R2:.4f}$$')

        x_min = st.number_input('Valor mínimo x:',value=X.min())
        x_max = st.number_input('Valor máximo x:',value=X.max()) 
        x_range_prediction = np.arange(x_min, x_max,1)
        log_y_range_prediction = model_exponential.predict(x_range_prediction.reshape(-1,1))
         
        plt.subplots()
        plt.title('Diagrama de dispersión y curva de regresión exponencial')
        plt.scatter(X, y)
        plt.plot(x_range_prediction, np.exp(log_y_range_prediction), color='red')
        plt.xlabel(X.name)
        plt.ylabel(y.name)
        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Predict a new value
        st.write('# Predicción de valores con el modelo de regresión exponencial')
        st.write('x: ',X.name)
        input_value = st.number_input(f'Introduce número de años después o antes de {X_min}', value=X.min(), key=next(widget_id))
        log_predicted_value = model_exponential.predict([[input_value]])
        st.write(f'En el año {X_min+input_value}, se tiene que {y.name} es {np.exp(log_predicted_value[0]):.2f}')
      
    else:
      X = filtered_data.iloc[:,0]
      y = filtered_data.iloc[:,1]
      log_y = np.log(y)
      log_X = np.log(X)   
      
      if st.checkbox('Diagramas de dispersión', key=next(widget_id)):
        st.write('# Diagramas de dispersión')
        fig, axes = plt.subplots(3, 1, figsize=(10,7))
    
        axes[0].set_title(f'Diagrama de dispersión - '+selected_dataset_2)
        axes[0].scatter(X,y)
        axes[0].set_xlabel(X.name)
        axes[0].set_ylabel(y.name)

        axes[1].set_title(f'Gráfica semi-log - '+selected_dataset_2)
        axes[1].scatter(X,log_y)
        axes[1].set_xlabel(X.name)
        axes[1].set_ylabel('Log '+y.name)

        axes[2].set_title(f'Gráfica log-log - '+selected_dataset_2)
        axes[2].scatter(log_X,log_y)
        axes[2].set_xlabel('Log '+X.name)
        axes[2].set_ylabel('Log '+y.name)

        fig.tight_layout()
    
        # Display the plot in Streamlit
        st.pyplot(fig)

      if st.checkbox('Mostrar modelo de regresión exponencial', key=next(widget_id)):
        model_exponential = LinearRegression()
        model_exponential.fit(X.values.reshape(-1,1), log_y)

        log_C = model_exponential.intercept_
        C = np.exp(log_C)
        k = model_exponential.coef_[0]
    
        st.latex(r'''y = Ce^{kx}  ''')
        st.write('Linearización')
        st.latex(r'''\ln(y) = kx + \ln(C) \quad \Rightarrow \quad Y_{\text{exp}} = kx + A_{\text{exp}}''')
        st.write('donde')
        st.latex(r'''Y_{\text{exp}}=\ln(y),\quad \text{y}\quad A_{\text{exp}}=\ln(C)''')
        st.write('Parámetros:')
        st.latex(r'''k='''+ rf'''{k:.4f}''')
        st.latex(r'''A_{\text{exp}}=\ln(C)='''+ rf'''{log_C:.4f}''' + r'''\quad \Rightarrow \quad C=''' + rf'''{C:.4f}''')

        st.write(f"Modelo de regresión exponencial: $$y = {C:.4f}x^{{{k:.4f}}}$$")
        log_y_predict = model_exponential.predict(X.values.reshape(-1,1))
        R2 = r2_score(log_y,log_y_predict)
        st.write(f'Coeficiente de determinación: $$R^2={R2:.4f}$$')

        x_min = st.number_input('Valor mínimo x:',value=X.min())
        x_max = st.number_input('Valor máximo x:',value=X.max()) 
        x_range_prediction = np.arange(x_min, x_max,1)
        log_y_range_prediction = model_exponential.predict(x_range_prediction.reshape(-1,1))
         
        plt.subplots()
        plt.title('Diagrama de dispersión y curva de regresión exponencial')
        plt.scatter(X, y)
        plt.plot(x_range_prediction, np.exp(log_y_range_prediction), color='red')
        plt.xlabel(X.name)
        plt.ylabel(y.name)
        # Display the plot in Streamlit
        st.pyplot(plt)
        
        # Predict a new value
        st.write('# Predicción de valores con el modelo de regresión exponencial')
        st.write('x: ',X.name)
        input_value = st.number_input(f'Introduce un valor de X', value=X.min(), key=next(widget_id))
        log_predicted_value = model_exponential.predict([[input_value]])
        st.write(f'Si {X.name} es {input_value}, entonces {y.name} es {np.exp(log_predicted_value[0]):.2f}')
      
      if st.checkbox('Mostrar modelo de regresión potencia', key=next(widget_id)):
        model_potential = LinearRegression()
        model_potential.fit(log_X.values.reshape(-1,1), log_y)

        log_a = model_potential.intercept_
        a = np.exp(log_a)
        n = model_potential.coef_[0]

        st.latex(r'''y = ax^{n}  ''')
        st.write('Linearización')
        st.latex(r''' \ln(y) = n\ln(x) + \ln(a) \quad \Rightarrow \quad Y_{\text{pot}} = nX_{\text{pot}} + A_{\text{pot}}''')
        st.write('donde')
        st.latex(r'''Y_{\text{pot}}=\ln(y),\quad X_{\text{pot}}=\ln(x), \quad \text{y}\quad A_{\text{pot}}=\ln(a)''')
        st.write('Parámetros:')
        st.latex(r'''n='''+ rf'''{n:.4f}''')
        st.latex(r'''A_{\text{pot}}=\ln(a)='''+ rf'''{log_a:.4f}''' + r'''\quad \Rightarrow \quad a=''' + rf'''{a:.6f}''')

        st.write(f"Modelo de regresión potencia: $$y = {a:.6f}x^{{{n:.4f}}}$$")
        log_y_predict = model_potential.predict(log_X.values.reshape(-1,1))
        R2 = r2_score(log_y,log_y_predict)
        st.write(f'Coeficiente de determinación: $$R^2={R2:.4f}$$')

        x_min = st.number_input('Valor mínimo x:',value=X.min(), key=next(widget_id))
        x_max = st.number_input('Valor máximo x:',value=X.max(), key=next(widget_id)) 
        log_x_range_prediction = np.arange(np.log(x_min), np.log(x_max),0.01)
        log_y_range_prediction = model_potential.predict(log_x_range_prediction.reshape(-1,1))
         
        plt.subplots()
        plt.title('Diagrama de dispersión y curva de regresión potencia')
        plt.scatter(X, y)
        plt.plot(np.exp(log_x_range_prediction), np.exp(log_y_range_prediction), color='red')
        plt.xlabel(X.name)
        plt.ylabel(y.name)
        # Display the plot in Streamlit
        st.pyplot(plt)

        # Predict a new value
        st.write('# Predicción de valores con el modelo de regresión potencia')
        st.write('x: ',X.name)
        input_value = st.number_input(f'Introduce un valor de X', value=X.min(), key=next(widget_id))
        log_predicted_value = model_potential.predict([[np.log(input_value)]])
        st.write(f'Si {X.name} es {input_value}, entonces {y.name} es {np.exp(log_predicted_value[0]):.2f}')

with tabs[2]:
  st.write('# Cargar base de datos')
  df_3 = pd.read_csv('Datasets/crecimiento_logistico.csv')

  # Display the Dataframe
  if st.checkbox('Mostrar base de datos', key=next(widget_id)):
    st.write('Base de datos: crecimiento_logistico')
    st.dataframe(df_3)

  if st.checkbox('Mostrar estadísticos descriptivos', key=next(widget_id)):
    st.write(df_3.describe())

  if st.checkbox('Diagrama de dispersión', key=next(widget_id)):
    st.write('# Diagrama de dispersión')
    plt.subplots()
    plt.title('Diagrama de dispersión - crecimiento_logistico')
    plt.scatter(df_3['Tiempo_dias'],df_3['Numero_moscas'])
    plt.xlabel(df_3['Tiempo_dias'].name)
    plt.ylabel(df_3['Numero_moscas'].name)
    # Display the plot in Streamlit
    st.pyplot(plt)
    
  if st.checkbox('Mostrar modelo de regresión logística', key=next(widget_id)):
    # Define logistic function
    def logistic_function(t, C, a, r):
      return  (C / (1 + a*np.exp(-r*t)))

    # fitting
    popt, _ = curve_fit(logistic_function, df_3['Tiempo_dias'], df_3['Numero_moscas'], p0=[df_3.max(), 1, 1], maxfev=1000)
    C = popt[0]
    a = popt[1]
    r = popt[2]
    st.write(f"Modelo de crecimiento logístico: $$y =\frac{C}{1 + a e^{{rt}}} = \frac{{C:.4f}}{1+a e^{{{r:.4f}t}}}$$")
        
    x_min = st.number_input('Valor mínimo x:',value=df_3['Tiempo_dias'].min(), key=next(widget_id))
    x_max = st.number_input('Valor máximo x:',value=df_3['Tiempo_dias'].max(), key=next(widget_id)) 
    # Generate predictions
    x_pred = np.linspace(x_min, x_max, 100)  # Time points for prediction
    y_pred = logistic_function(x_pred, C, a, r)
         
    plt.subplots()
    plt.title('Diagrama de dispersión y curva de regresión logística')   
    plt.scatter(df_3['Tiempo_dias'], df_3['Numero_moscas'])
    plt.plot(x_pred, y_pred, color='red')
    plt.xlabel(df_3['Tiempo_dias'].name)
    plt.ylabel(df_3['Numero_moscas'].name)
    # Display the plot in Streamlit
    st.pyplot(plt)
