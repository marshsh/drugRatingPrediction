
# Instalación Previa

Sumponemos que ya está instalado:

SMH
nltk

# Observaciones

**Ver archivo Resultados.**

# Datos Importantes

En el conjunto de TRAIN se encontraron 36414, en el vocabulario (.vocab) (lematizadas y sin StopWords)
De ellas, en el (.ifs) nos quedamos con 36278 palabras (lematizadas y sin StopWords)


TRAIN :
(.tsv RAW) 209991
(.ref4train) 143316
(.ref) 
(.labels) 143316
(train_rate.pickle) 286636
(train_reviews.RAW_pickle) 286636

(.corpus) 179282

(.vocab) 38247
(.ifs) 36278

topics: 1779
(.words2topics) 38241 .... no importa que no coincida con (.vocab) eso sólo significa que no hay tópicos con las palabras que faltaron, y por eso se perdieron en el `smhcmd ifindex`, pero el númeroID de la palabra sigue siendo el renglón en el que se encuentra.


TEST
(.tsv RAW) 70491
(.ref) 47699
(.labels) 47703
(test_rate.pickle) 95400
(test_reviews.RAW_pickle) 95400

********* Había un error en ref2corpus ... min_doc_terms estaba con default = 1


# Archivos intermedios

_.labels_ son los valores de 'rating' (que queremos predecir)

_.ref_ son las reseñas lemmatizadas (pero con todo y stopWords)

_.vocab_ y _.corpus_ se construyen al mismo tiempo

_.vocab_ tiene el formato " palabra = token = freq1 freq2 "

_.corpus_ cada renglón es una reseña ... tiene el formato de listdb: " sizeReseña tokenP:freqP tokenP:freqP tokenP:freqP tokenP:freqP" 

_.models_ cada renglón es un tópico ... tiene el formato de listdb: " sizeTopic tokenP:freqP tokenP:freqP tokenP:freqP tokenP:freqP"  donde freqP es la frecuencia de la palabra en los COWS


_.ifs_ Se obtiene a partir del .corpus ... en él, cada documento es llamado por el número de renglón en el que aparece en .corpus, y depues del #documento se observa la frecuencia con la que aparece la palabra en dicho documento. Las palabras conservan el mismo nombre-token que en el .corpus (osea que tambien en el .vocab)


## Nuevos Archivos

_.orderes\_models_ es una listdb [ size (topicWordNum\_i : freq\_i)\_{for i in size} ] pero cuyos topics están ordenados de acuerdo al número de documentos que contienen alguna de sus palabras

_.orderes\_models\_words_ es la versión explícita de _.orderes\_models_. Se estructura así:
(número de documentos associados al tópico) ... size (word\_i : freq\_i)\_{for i in size}


