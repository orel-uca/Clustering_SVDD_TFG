# ClusterSVDD

Este proyecto contiene una implementación en Python de distintos métodos de clasificación y detección de anomalías, con especial atención a la Descripción de Datos mediante Vectores de Soporte (Support Vector Data Description, SVDD) y a su extensión multiesfera ClusterSVDD. El proyecto es parte del **TFG de Javier Rrmero Hurtado**, alumno del **Grado en Matemáticas de la Universidad de Cádiz**.

Se incluyen los modelos primal y dual de SVDD, ClusterSVDD, que combina la construcción de múltiples descripciones SVDD con un procedimiento iterativo de clustering y One-Class SVM (OC-SVM). Además el código permite generar diferentes conjuntos de datos sintéticos, aplicar ponderación por densidad y evaluar los modelos mediante distintas métricas de clasificación.

# CARACTERÍSTICAS PRINCIPALES

* SVDD: Implementación de la formulación primal de SVDD mediante Gurobi.
* ClusterSVDD: Extensión multiesfera de SVDD. El procedimiento comienza con una partición obtenida mediante k-means y alterna entre el ajuste de una SVDD para cada grupo y la reasignación de las observaciones a las distintas esferas.
* One-Class SVM: Implementación mediante scikit-learn para realizar comparaciones con SVDD.
* Ponderación por densidad: Posibilidad de utilizar pesos dependientes de la densidad local de las observaciones durante el ajuste de SVDD y ClusterSVDD.
* Funciones kernel: Soporte para kernels lineales y no lineales, incluyendo el kernel RBF.
* Generación de datos: Scripts para crear conjuntos de datos sintéticos en 2D con una fracción de anomalías configurable.. Se consideran, entre otras, estructuras esféricas, elípticas y no lineales.
* Estructuras multimodales: Posibilidad de generar datos regulares distribuidos en varios grupos mediante el parámetro data_k.
* Evaluación de modelos: Cálculo de métricas de rendimiento como AUC-ROC y matrices de confusión.
* Validación cruzada: Selección y comparación de parámetros a partir de los resultados obtenidos sobre el conjunto de validación.
* Visualización: Funciones para graficar las fronteras de decisión, los datos de entrenamiento/test y los outliers detectados.
* Experimentación: Un script principal parametrizable (`test_SVDD.ipynb`) que permite configurar y ejecutar experimentos fácilmente.

# ESTRUCTURA DEL PROYECTO

* models.py
  Contiene la implementación de los modelos principales:
  - Primal_SVDD
  - Dual_SVDD
  - Cluster_SVDD
  - Sklearn_SVM
* svdd_class.py
  Define las clases utilizadas para representar los problemas y almacenar las soluciones:
  - Instance
  - Solution
  - SklearnSolution
  - MultiSphereSolution
* utilities.py
  Contiene las funciones auxiliares utilizadas a lo largo del proyecto, entre ellas:
  - cálculo de kernels
  - cálculo de scores
  - generación de datos sintéticos
  - cálculo de pesos basados en densidad
  - entrenamiento de las distintas SVDD de ClusterSVDD
  - evaluación de modelos
  - cálculo de métricas
  - validación cruzada
  - generación de gráficos
* main.py
  Script principal del proyecto. Procesa los argumentos de entrada y coordina la generación de datos, el entrenamiento de los modelos, el cálculo de métricas y la validación cruzada.
* ejemplos.ipynb
  Notebook de Jupyter con ejemplos de los experimentos para facilitar la modificación de sus parámetros.
* Data/: Directorio destinado a almacenar los conjuntos de datos generados y utilizados en los experimentos.
* RDB/: Directorio con los datos de bases de datos reales utilizados en los experimentos.
* Salidas_.../: Directorios donde se guardan los resultados de los experimentos, como métricas y logs.
* README.md: Este archivo.

# REQUISITOS

El proyecto requiere Python y las siguientes bibliotecas:

* NumPy
* SciPy
* Matplotlib
* scikit-learn
* gurobipy

# ARGUMENTOS DEL SCRIPT

--new_data: Genera nuevos conjuntos de datos sintéticos antes de realizar el experimento.
--dir_data: Directorio en el que se almacenan o desde el que se cargan los conjuntos de datos.
--dir_sufix_out: Sufijo utilizado para distinguir los directorios y archivos de salida correspondientes a diferentes experimentos.
--method: Método utilizado en el experimento. Las principales opciones son:
  svdd
  clustersvdd
  ocsvm
--use_kernel: Activa el uso de kernel. Para SVDD y OCSVM se utiliza la formulación dual en lugar de la formulación primal.
--use_density: Activa la ponderación de las observaciones en función de su densidad local.
--do_evaluation: Entrena los modelos para las configuraciones especificadas.
--do_metrics: Calcula y almacena las métricas de rendimiento obtenidas por los modelos.
--do_cross_val: Realiza la comparación de los resultados de validación para los distintos valores de los parámetros considerados.
--show_plot_detail: Muestra las figuras generadas durante la evaluación.
--save_plot_detail: Guarda las figuras generadas durante los experimentos.
--reps: Identificadores de las repeticiones independientes que se desean ejecutar.
--Nus: Lista de valores considerados para el parámetro de regularización nu.
--sigmas: Lista de valores considerados para el parámetro del kernel RBF. Este argumento es necesario cuando se utiliza la formulación con kernel.
--num_train: Número de observaciones utilizadas para entrenamiento.
--num_val: Número de observaciones utilizadas para validación.
--num_test: Número de observaciones utilizadas para test.
--anom_frac: Fracción de anomalías presente en los conjuntos de datos generados. Las opciones son:
  spherical
  elliptical
  banana
--data_shape: Geometría de los datos sintéticos utilizados en el experimento.
--data_k: Número de grupos o componentes utilizados para generar los datos normales.
--k: Número de esferas utilizadas por ClusterSVDD. Para k=1, ClusterSVDD se reduce a una única descripción SVDD.
--max_iter: Número máximo de iteraciones permitido en el procedimiento iterativo de ClusterSVDD.
--random_state: Semilla utilizada para controlar la aleatoriedad y poder reproducir de los experimentos.

# EJECUCIÓN

Los experimentos pueden ejecutarse desde la línea de comandos mediante main.py:

Ejemplo:

python main.py
--new_data
--dir_data Data
--dir_sufix_out test
--do_evaluation
--do_metrics
--do_cross_val
--reps 1 2 3
--Nus 0.05 0.10
--num_train 100
--num_val 66
--num_test 166
--anom_frac 0.10
--data_shape spherical
--method clustersvdd
--data_k 2
--k 2

También puede ejecutarse desde Jupiter Notebook llamando al script main.py con %run:

%run main.py
--new_data
--dir_data Data
--dir_sufix_out test
--do_evaluation
--do_metrics
--reps 1 2 3
--Nus 0.05 0.10
--num_train 100
--num_val 66
--num_test 166
--anom_frac 0.10
--data_shape spherical
--method clustersvdd
--data_k 2
--k 2
