# Discusión de Resultados
=================================

Obtuvimos los siguientes resultados:

|Modelo| Loss |Test MAE | Test Accuracy (1 point tolerance)|
|--|--|--|--|
|SMH|0.185369491577|2.63241887093|0.355255752802|
|BOW|2.6387441721  |2.6387441721 |0.340628023113|



## Como correr el programa

sh prepare_drugs.sh
python ./python/train/trainReviews.py <emb> -cB
python ./python/train/evalReviews.py <emb> mae


Dónde <emb> puede tomar los valores "SMH" o "BOW" para representar las reseñas usando vectores de tópicos generados con SMH, o vectores que representen al documento como una bolsa de palabras (BOW).

## Procesamiento del corpus

Para procesar el corpus, lemmatizamos todos los textos, quitamos signos de puntuación y palabras `stopwords`.

## Descubrimiento de Tópicos

Utilizamos la herramienta Sampled Min-Hash para descubrir los tópicos del corpus. Utilizamos únicamente tuplas de tamaño 2. Aunque dada la estructura del programa, sería muy fácil realizar el experimento de obserbar si aumenta el rendimiento con un tamaño de tupla mayor, por el momento nos quedamos tópicos descubiertos con tuplas de tamaño 2.

Es importnte obserbar, que no todas las reseñas del corpus tienen un rating asosiado. Aún así, dichas reseñas nos sirven para entrenar la herramienta SMH de descurimiento de tópicos.

También quiero mencionar que la fracción del corpus de prueba (TEST) no se utilizó para entrenar a SMH.

## Red Neuronal

Utilizamos una red extremadamente sencilla, diseñada para una tarea de regresión:

|Input(Embedding Size)|
|--|
|Dense (128)|
|Activation = ELU|
|Dropout (0.5)|
|Dense (64)|
|Activation = ELU|
|Dropout (0.5)|
|Dense (16)|
|Activation = ELU|
|Dropout (0.5)|
|Dense (1)|


## Encajes de Documentos

Aquí decidí bifurcar el modelo.

* Por un lado representamos al documento con un vector a través de su BOW (Bolsa de Palabras). Cada entrada representa cuántas veces aparece en el documento la palabra correspondiente. 
	* Estos vectores son dispersos, y tienen una longitud del tamaño del vocabulario. 38247 en este caso.

* Por otro lado, tenemos una representación basada en SMH. Cada entrada del vector representa un tópico distinto. Y el valor de la entrada representa que tanto aparecen las palabras del tópico correspondiente en el documento. 
	* Par ser específicos: las palabras de los tópicos tienen un valor de importancia. Multiplicamos la importancia de cada palara del tópico por la cantidad de veces que aparece en el documento, sumamos sobre todas las palabras, y ese es el valor de la entrada. 
	* La longitud de estos vectores es relativamente pequeña, sobretodo en comparación con los vectores de BOW. 1779 en este caso.

## Entrenamiento

Entrenamos la red neuronal con función de pérdida `mean_absolute_error` y optimizador `sgd` (Stochastic Gradient Descent). Le pedimos al modelo que calculara las métricas `mae` y `soft_acc`.

Originalmente empezamos con función de pérdida `mean_squared_error`, pero decidimos cambiar debido a que nos interesa mucho que se reduzcan los errores pequeños, pues estos marcan la diferencia entre que el *rating* predicho se redondee hacia el *rating* verdadero o en sentido contrario (abajo o arriba).

`soft_acc` es una función definida por el usuario (nosotros) que calcula a cuántos ratings le atinamos si redondeamos el valor de regresión que predice nuestro modelo.


## Evaluación

El principal elemento de evaluación, dada la naturaleza del problema, es la *exactitud* (accuracy). Pero tenemos distintas formas de medirla.

En las evaluaciones mostradas a continuación y al inicio del documento, tenemos 3 medidas Loss, Test MAE y Test Accuracy (1 point tolerance). 

La más representativa e interpretable es Test Accuracy (1 point tolerance).

Mide qué porcentaje de las reseñas predichas, (redondeadas al entero más cercano), acertaron o se quedaron a 1 punto de distancia.

0.36 para vectores SMH me parece un buen resultado.
El azhar nos otorga un 0.25.
Los vectores BOW obtuvieron una exactitud de 0.34, también muy buena.


* Estos resultados nos dicen que las reseñas si contienen informción importante relacionada con el *rating* que deciden los usuarios.
* También podemos obserbar, que estos resultados nos indican que los vectores SMH son una buena forma de coprimir la información de documetos. (Por lo menos para éste ejemplo). 
	* Es una reducción de 38247 a 1779. Es decir, una reducción de escala 20:1 , y además, el rendimiento no sólo no se redujo si no que mejoró ligeramente. No mucho, tal vez debido al azhar, pero subió un 0.015 en la exactitud.



|Modelo| Loss |Test MAE | Test Accuracy (1 point tolerance)|
|--|--|--|--|
|SMH|0.185369491577|2.63241887093|0.355255752802|
|BOW|2.6387441721  |2.6387441721 |0.340628023113|


* En mi opinión, **debido a la arquitectura de la red neuronal utilizada**, los resultados obtenidos en este caso podrían no deberse a la capacidad de los tópicos y las bolsas palabras de destilar el sentimiento de las reseñas. Según mi percepción, los resultados relativamente positivos obtenidos, podrían deberse a un fenómeno de clasificación, en donde las palabras utilizadas le indican al modelo el típo de droga utilizada, y cada tipo de droga seguramente tiene su distribución de _rating_ asosiada. 

Eso podría explicar los resultados mayores al azhar.

Es muy importante notar que hacemos esta observacion, porque la red neuronal utilizada no toma en cuenta el orden en el que aparecen las palabras. Por lo que una reseña que exprese que "No tiene tales síntomas", al quitar los _stopwords_ y revolver el orden de palabras, puede quedar igual que una reseña que expresa que "Si tiene tales síntomas".

Un mejor modelo requeriría que la aruitectura de la red neuronal incluyera una capa LSTM Long Short Term Memmory.

Si se quieren observar los tópicos recuperados por el modelo, se puede abrir el archivo "./drugsCom_SMH/smh\_...\_.ordered_models_words"


## Conclusión

Para corpus similares a drugsCom, el descubrimiento de tópicos a través de SMH es una excelente herramienta de reducción de dimensionalidad.

