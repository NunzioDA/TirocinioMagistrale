import tensorflow as tf # type: ignore
import numpy as np
from keras import activations # type: ignore
from keras import initializers # type: ignore

class ConditionsLayer(tf.keras.layers.Layer):
    def __init__(
            self, 
            units,
            fixed_w1=None,
            fixed_w2=None,
            greather_fixed_weight=None,
            smaller_fixed_weight=None,
            initializer="glorot_uniform",
            activation="relu",
        ):
        super(ConditionsLayer, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.initializer_w1 = tf.keras.initializers.RandomUniform()
        self.initializer_w2 = tf.keras.initializers.RandomUniform()
        self.fixed_w1 = fixed_w1
        self.fixed_w2 = fixed_w2
        self.greater_fixed_weight = greather_fixed_weight
        self.smaller_fixed_weight = smaller_fixed_weight

    def get_mask(self, fixed_weights):
        mask = tf.equal(fixed_weights, -1)
        mask = tf.cast(mask, dtype=tf.float32)
        # mask = tf.constant(mask, dtype=tf.float32)
        return mask

    def assign_fixed(self, weights, fixed_weights):
        mask = self.get_mask( fixed_weights)
        no_nan_weights = tf.where(fixed_weights == -1, tf.zeros_like(fixed_weights), fixed_weights)
        weights.assign(weights * mask + no_nan_weights)
    
    def broadcast_sum(self, M1,M2):
        # M1 = pxm
        # M2 = mxn
        # result = pxn

        # Estendiamo matrix1 e matrix2 per effettuare l'operazione di broadcasting
        matrix1_expanded = tf.expand_dims(M1, axis=-1)  # pxm x 1
        matrix1_expanded = tf.tile(matrix1_expanded, [1, 1, M2.shape[-1]])  # pxm x 1 -> pxm x n

        result = matrix1_expanded + tf.expand_dims(M2, axis=0)  # pxm x n + 1 x mxn

        return result

    def build(self, input_shape):
        # Inizializzo i pesi w1 e w2
        self.w1 = self.add_weight(shape=(self.units,), initializer=self.initializer_w1, trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(self.units,), initializer=self.initializer_w2, trainable=True, name='w2')

        greather_conditions_weight = np.array([[0 for _ in range(self.units)] for _ in range(input_shape[1])])
        smaller_conditions_weight = np.array([[0 for _ in range(self.units)] for _ in range(input_shape[1])])
        for i in range(0, self.units):
            if(((i) // input_shape[1])%2==0):
                greather_conditions_weight[(i) % input_shape[1] ][i] = 1
            else: 
                smaller_conditions_weight[(i) % input_shape[1] ][i] = 1
        self.greater_weight = tf.convert_to_tensor(greather_conditions_weight, dtype=tf.float32)
        self.smaller_weight = tf.convert_to_tensor(smaller_conditions_weight, dtype=tf.float32)
        
    def call(self, inputs):
        if self.units > 0:
            activated_w1_minus_f = self.activation(self.broadcast_sum(inputs, -(self.greater_weight * self.w1)))
            activated_f_minus_w2 = self.activation(self.broadcast_sum(-inputs, (self.smaller_weight * self.w2)))


            w1_minus_f = tf.reduce_sum(self.greater_weight * activated_w1_minus_f, axis=1)
            f_minus_w2 = tf.reduce_sum(self.smaller_weight * activated_f_minus_w2, axis=1)
            output = w1_minus_f + f_minus_w2

        if self.fixed_w1 is not None and self.fixed_w2 is not None:
            activated_w1_minus_f_fixed = self.activation(self.broadcast_sum(inputs, -(self.greater_fixed_weight * self.fixed_w1)))
            activated_f_minus_w2_fixed = self.activation(self.broadcast_sum(-inputs, (self.smaller_fixed_weight * self.fixed_w2)))


            w1_minus_f_fixed = tf.reduce_sum(self.greater_fixed_weight * activated_w1_minus_f_fixed, axis=1)
            f_minus_w2_fixed = tf.reduce_sum(self.smaller_fixed_weight * activated_f_minus_w2_fixed, axis=1)

            output_fixed = w1_minus_f_fixed + f_minus_w2_fixed
            result = tf.concat([output_fixed, output],axis=1)
        else:
            output_fixed = tf.constant([], dtype=tf.float32)
            result = output

        # Somma i termini e restituisci il risultato finale
        
        
        # Ritorna la somma lungo l'asse delle feature
        return result
    

class ConditionsLayer3(tf.keras.layers.Layer):
    def __init__(
            self, 
            units,
            fixed_w1=None,
            fixed_w2=None,
            greather_fixed_weight=None,
            smaller_fixed_weight=None,
            initializer="glorot_uniform",
            activation="sigmoid",
        ):
        super(ConditionsLayer, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.relu = activations.get("relu")
        self.sigmoid = activations.get("sigmoid")
        self.initializer_w1 = tf.keras.initializers.RandomUniform()
        self.initializer_w2 = tf.keras.initializers.RandomUniform()
        self.fixed_w1 = fixed_w1
        self.fixed_w2 = fixed_w2
        self.greater_weight = greather_fixed_weight
        self.smaller_weight = smaller_fixed_weight

    def get_mask(self, fixed_weights):
        mask = tf.equal(fixed_weights, -1)
        mask = tf.cast(mask, dtype=tf.float32)
        # mask = tf.constant(mask, dtype=tf.float32)
        return mask

    def assign_fixed(self, weights, fixed_weights):
        mask = self.get_mask( fixed_weights)
        no_nan_weights = tf.where(fixed_weights == -1, tf.zeros_like(fixed_weights), fixed_weights)
        weights.assign(weights * mask + no_nan_weights)
    
    def broadcast_sum(self, M1,M2):
        # M1 = pxm
        # M2 = mxn
        # result = pxn

        # Estendiamo matrix1 e matrix2 per effettuare l'operazione di broadcasting
        matrix1_expanded = tf.expand_dims(M1, axis=-1)  # pxm x 1
        matrix1_expanded = tf.tile(matrix1_expanded, [1, 1, M2.shape[-1]])  # pxm x 1 -> pxm x n

        result = matrix1_expanded + tf.expand_dims(M2, axis=0)  # pxm x n + 1 x mxn

        return result

    def build(self, input_shape):
        # Inizializzo i pesi w1 e w2
        self.w1 = self.add_weight(shape=(self.units,), initializer=self.initializer_w1, trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(self.units,), initializer=self.initializer_w2, trainable=True, name='w2')

        if self.fixed_w1 is not None:
            self.assign_fixed(self.w1, self.fixed_w1)
        if self.fixed_w2 is not None:
            self.assign_fixed(self.w2, self.fixed_w2)

        
    def call(self, inputs):

        if self.fixed_w1 is not None:
            self.assign_fixed(self.w1, self.fixed_w1)
        if self.fixed_w2 is not None:
            self.assign_fixed(self.w2, self.fixed_w2)


        activated_w1_minus_f = self.relu(self.broadcast_sum(inputs, -(self.greater_weight * self.w1)))
        activated_f_minus_w2 = self.relu(self.broadcast_sum(-inputs, (self.smaller_weight * self.w2)))


        w1_minus_f = tf.reduce_sum(self.greater_weight * activated_w1_minus_f, axis=1)
        f_minus_w2 = tf.reduce_sum(self.smaller_weight * activated_f_minus_w2, axis=1)

        # Somma i termini e restituisci il risultato finale
        output = w1_minus_f + f_minus_w2
        
        # Ritorna la somma lungo l'asse delle feature
        return output

class ConditionsLayer2(tf.keras.layers.Layer):
    def __init__(
            self, 
            units,
            fixed_w1=None,
            fixed_w2=None,
            greather_fixed_weight=None,
            smaller_fixed_weight=None,
            initializer="glorot_uniform",
            activation="sigmoid",
        ):
        super(ConditionsLayer, self).__init__()
        self.units = units
        self.activation = activations.get(activation)
        self.relu = activations.get("relu")
        self.sigmoid = activations.get("sigmoid")
        self.initializer_w1 = tf.keras.initializers.RandomUniform(minval=50, maxval=150)
        self.initializer_w2 = tf.keras.initializers.RandomUniform(minval=50, maxval=150)
        self.fixed_w1 = fixed_w1
        self.fixed_w2 = fixed_w2
        self.greather_fixed_weight = greather_fixed_weight
        self.smaller_fixed_weight = smaller_fixed_weight

    def get_mask(self, fixed_weights):
        mask = tf.equal(fixed_weights, -1)
        mask = tf.cast(mask, dtype=tf.float32)
        # mask = tf.constant(mask, dtype=tf.float32)
        return mask

    def assign_fixed(self, weights, fixed_weights):
        mask = self.get_mask( fixed_weights)
        no_nan_weights = tf.where(fixed_weights == -1, tf.zeros_like(fixed_weights), fixed_weights)
        weights.assign(weights * mask + no_nan_weights)
    
    def broadcast_sum(self, M1,M2):
        # M1 = pxm
        # M2 = mxn
        # result = pxn

        # Estendiamo matrix1 e matrix2 per effettuare l'operazione di broadcasting
        matrix1_expanded = tf.expand_dims(M1, axis=-1)  # pxm x 1
        matrix1_expanded = tf.tile(matrix1_expanded, [1, 1, M2.shape[-1]])  # pxm x 1 -> pxm x n

        result = matrix1_expanded + tf.expand_dims(M2, axis=0)  # pxm x n + 1 x mxn

        return result

    def build(self, input_shape):
        # Inizializzo i pesi w1 e w2
        self.w1 = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.initializer_w1, trainable=True, name='w1')
        self.w2 = self.add_weight(shape=(input_shape[-1], self.units), initializer=self.initializer_w2, trainable=True, name='w2')
        self.greater_weight = self.add_weight(shape=(1,input_shape[-1], self.units), initializer=self.initializer_w3, trainable=True, name='greater_weight')
        self.smaller_weight = self.add_weight(shape=(1,input_shape[-1], self.units), initializer=self.initializer_w4, trainable=True, name='smaller_weight')

        if(self.greather_fixed_weight is not None):
            self.greather_fixed_weight = tf.expand_dims(self.greather_fixed_weight, axis=0)

        if(self.smaller_fixed_weight is not None):
            self.smaller_fixed_weight = tf.expand_dims(self.smaller_fixed_weight, axis=0)

        if self.fixed_w1 is not None:
            self.assign_fixed(self.w1, self.fixed_w1)
        if self.fixed_w2 is not None:
            self.assign_fixed(self.w2, self.fixed_w2)

        if self.greather_fixed_weight is not None:
            self.assign_fixed(self.greater_weight, self.greather_fixed_weight)
        if self.smaller_fixed_weight is not None:
            self.assign_fixed(self.smaller_weight, self.smaller_fixed_weight)
        # self.add_loss(lambda: self.custom_regularization())

    # def custom_regularization2(self):


    def custom_regularization(self):
        # Penalizza la coesistenza di valori non nulli in w1 e w2

        regularization_penalty_greater_weight = tf.where(
            self.greater_weight  < 0, 
            tf.zeros_like(self.greater_weight), 
            self.greater_weight
        ) / tf.abs(self.greater_weight)

        regularization_penalty_smaller_weight = tf.where(
            self.smaller_weight <0, 
            tf.zeros_like(self.smaller_weight), 
            self.smaller_weight
        ) / tf.abs(self.smaller_weight)

        regularization_penalty_greater_weight = tf.reduce_sum(regularization_penalty_greater_weight, axis=1)
        regularization_penalty_smaller_weight = tf.reduce_sum(regularization_penalty_smaller_weight, axis=1)

        regularization_penalty_greater_weight = tf.where(
            regularization_penalty_greater_weight <3, 
            tf.zeros_like(regularization_penalty_greater_weight), 
            regularization_penalty_greater_weight
        )

        regularization_penalty_greater_weight = tf.where(
            regularization_penalty_greater_weight == 0, 
            tf.ones_like(regularization_penalty_greater_weight), 
            regularization_penalty_greater_weight
        )

        regularization_penalty_smaller_weight = tf.where(
            regularization_penalty_smaller_weight <3, 
            tf.zeros_like(regularization_penalty_smaller_weight), 
            regularization_penalty_smaller_weight
        )
        regularization_penalty_smaller_weight = tf.where(
            regularization_penalty_smaller_weight == 0, 
            tf.ones_like(regularization_penalty_smaller_weight), 
            regularization_penalty_smaller_weight
        )

        regularization_penalty_greater_weight = tf.reduce_sum(regularization_penalty_greater_weight, axis=1)[0]
        regularization_penalty_smaller_weight = tf.reduce_sum(regularization_penalty_smaller_weight, axis=1)[0]

        return (regularization_penalty_greater_weight  +  regularization_penalty_smaller_weight) * 5

    def call(self, inputs):

        if self.fixed_w1 is not None:
            self.assign_fixed(self.w1, self.fixed_w1)
        if self.fixed_w2 is not None:
            self.assign_fixed(self.w2, self.fixed_w2)

        if self.greather_fixed_weight is not None:
            self.assign_fixed(self.greater_weight, self.greather_fixed_weight)
        if self.smaller_fixed_weight is not None:
            self.assign_fixed(self.smaller_weight, self.smaller_fixed_weight)

        # sign_greater = greather_sum / inputs
        # sign_smaller = smaller_sum / inputs

        activated_w1_minus_f = self.relu(self.broadcast_sum(inputs, -self.w1))
        activated_f_minus_w2 = self.relu(self.broadcast_sum(-inputs, self.w2))


        w1_minus_f = tf.reduce_sum(self.greater_weight * activated_w1_minus_f, axis=1)
        f_minus_w2 = tf.reduce_sum(self.smaller_weight * activated_f_minus_w2, axis=1)

        # Somma i termini e restituisci il risultato finale
        output = w1_minus_f + f_minus_w2
        
        # Ritorna la somma lungo l'asse delle feature
        return output
    
