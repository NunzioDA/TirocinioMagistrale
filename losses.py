
def contrastive_loss(y_true, y_pred, margin=2.0):
    # Converti y_true in float32 per compatibilità con y_pred

    y_true = tf.cast(y_true, tf.float32)
    
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

def custom_loss(y_true, y_pred, weights):
    # Definisci i centri per ogni classe
    centers = tf.constant([[0.0, 0.0,0.0], [1.0, 1.0, 1.0]], dtype=tf.float32)

    # Converti le etichette (y_true) in interi se necessario
    y_true = tf.cast(y_true, tf.int32)

    # Recupera i centri corrispondenti alle etichette vere
    class_centers = tf.gather(centers, y_true)

    # Calcola la distanza euclidea tra le predizioni e i centri corrispondenti
    loss = tf.reduce_mean(tf.square(y_pred - class_centers))

    return loss

def class_splitter_loss(y_true, y_pred):
    num_dimensions = tf.shape(y_pred)[1] 

    # Costruisci il centro per le classi
    center_0 = tf.zeros([num_dimensions], dtype=tf.float32)
    center_1 = tf.ones([num_dimensions], dtype=tf.float32)

    # Combina i centri in un tensore
    centers = tf.stack([center_0, center_1], axis=0)
    y_true = tf.cast(y_true, tf.int32)

    class_centers = tf.gather(centers, y_true)

    distances = tf.reduce_sum(tf.square(y_pred - class_centers), axis=1)

    loss = tf.reduce_mean(tf.square(y_pred - class_centers))

    return loss
def create_pairs(X, y, knowledge_elements_weight=2):
    pairs = []
    labels = []
    weights = []
    
    # Mappatura delle classi a indici posizionali
    digit_indices = {label: X.index[y == label].tolist() for label in y.unique()}  # Indici effettivi
    
    for idx in X.index:
        current_image = X.loc[idx].values  # Usa loc per accedere all'indice reale
        label = y.loc[idx]  # Ottieni l'etichetta corrispondente
        
        # Aggiungi un esempio positivo (stessa classe)
        positive_idx = np.random.choice(digit_indices[label])
        positive_image = X.loc[positive_idx].values  # Usa loc per l'indice reale
        pairs.append([current_image, positive_image])
        labels.append(1)
        
        
        # Aggiungi un esempio negativo (classe diversa)
        negative_label = np.random.choice(y[y != label].unique())  # Scegli una classe diversa
        negative_idx = np.random.choice(digit_indices[negative_label])
        negative_image = X.loc[negative_idx].values  # Usa loc per l'indice reale
        pairs.append([current_image, negative_image])
        labels.append(0)

        if(((X.loc[idx]["BMI"] >= 30) and (X.loc[idx]["Glucose"] >= 126)) or ((X.loc[idx]["BMI"] <= 25) and (X.loc[idx]["Glucose"] <= 100))):
            weights.append(knowledge_elements_weight)
            weights.append(knowledge_elements_weight)
        else:
            weights.append(1)
            weights.append(1)
    
    return np.array(pairs), np.array(labels), np.array(weights)