# custom_units = df.shape[1]*4
# conditions = 4
# learning_rate=0.00001

# fixed_greather_condition = [0 for _ in range(conditions)]

# fixed_greather_condition[0] = 126 / maxes[1]
# fixed_greather_condition[1] = 30 / maxes[5]

# fixed_greather_condition = tf.constant(fixed_greather_condition, dtype=tf.float32)


# greather_conditions_weight = np.array([[0 for _ in range(conditions)] for _ in range(df.shape[1])])
# greather_conditions_weight[1][0] = 1
# greather_conditions_weight[5][1] = 1


# fixed_smaller_condition = [0 for _ in range(conditions)]
# fixed_smaller_condition[2] = 100 / maxes[1]
# fixed_smaller_condition[3] = 25 / maxes[5]

# # fixed_smaller_condition[31] = -0.020948993
# # fixed_smaller_condition[17] = -0.002275765
# # fixed_smaller_condition[25] = 0.022
# fixed_smaller_condition = tf.constant(fixed_smaller_condition, dtype=tf.float32)

# smaller_conditions_weight = np.array([[0 for _ in range(conditions)] for _ in range(df.shape[1])])
# smaller_conditions_weight[1][2] = 1
# smaller_conditions_weight[5][3] = 1

# greather_conditions_weight = tf.convert_to_tensor(greather_conditions_weight, dtype=tf.float32)
# smaller_conditions_weight = tf.convert_to_tensor(smaller_conditions_weight, dtype=tf.float32)

##################################################################################

# class_distance = getClassAvgDistance(points, target)
# print(class_distance)
# indx=np.argsort(class_distance[fixed_greather_condition.shape[0]:])
# print(indx%8)
# print((indx//8)%2)
# print(model.layers[1].get_weights()[0][indx])
# print(model.layers[1].get_weights()[1][indx])

# fixed_greather_condition = fixed_greather_condition.numpy()
# fixed_smaller_condition = fixed_smaller_condition.numpy()


# for i in indx[-5:]:    
#     fixed_greather_condition = np.concatenate([fixed_greather_condition, np.array([model.layers[1].get_weights()[0][i]])])
#     fixed_smaller_condition = np.concatenate([fixed_smaller_condition, np.array([model.layers[1].get_weights()[1][i]])])

#     new_column = np.zeros_like(greather_conditions_weight[:,0][:,np.newaxis], dtype=np.float32)
#     greater_new_column=new_column.copy()
    

#     if((i//8)%2 == 0):        
#         greater_new_column[i%8][0]=1
#     else:
#         new_column[i%8][0]=1

#     greather_conditions_weight = np.concatenate([greather_conditions_weight, greater_new_column],axis=1)
#     smaller_conditions_weight = np.concatenate([smaller_conditions_weight, new_column],axis=1)

# fixed_smaller_condition = tf.constant(fixed_smaller_condition, dtype=tf.float32)
# fixed_greather_condition = tf.constant(fixed_greather_condition, dtype=tf.float32)


#########################################################################################

# from sklearn.ensemble import RandomForestClassifier

# Xp_train = m.predict(X_train)
# p_test = m.predict(X_test)

# # Inizializzazione del modello
# random_forest = RandomForestClassifier(random_state=42,)

# # Adattamento del modello sui dati di addestramento
# random_forest.fit(Xp_train, y_train)

# pred = random_forest.predict(p_test)

# # pred[(p_test["BMI"] >= 30) & (p_test["Glucose"] >= 126)] = 1
# # pred[(p_test["BMI"] <= 25) & (p_test["Glucose"] <= 100)] = 0

# accuracy_score(y_test, pred)

########################################################################
# # points = base_model.predict(X_train)
# # print(points.shape)

# # plt.scatter(points[:,0],points[:,1], c=y_train)
# m = keras.Model(inputs=input_layer, outputs=custom_layer)



# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=learning_rate,
# )

# m.compile(
#     # optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
#     # loss = lambda y_true, y_pred : class_splitter_loss(y_true, y_pred),
#     optimizer=optimizer,
#     loss="binary_crossentropy", 
#     metrics=['accuracy']
# )

# points = m.predict(df.to_numpy())
# print(points)
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# t = target.copy()
# # cf = pd.read_csv('diabetes.csv')
# t[(df["BMI"] >= 30 / maxes[5]) & (df["Glucose"] >= 126/ maxes[1])]=2
# t[(df["BMI"] >= 30 / maxes[5]) & (df["Glucose"] >= 126/ maxes[1]) & (target == 1)]=3

# # Definisci un dizionario di colori per ogni etichetta
# color_map = {0: 'green', 1: 'red', 2: 'yellow', 3: 'purple'}

# # Crea una lista di colori basata sulle etichette
# colors = [color_map[label] for label in t]

# # Scatter plot 3D
# ax.scatter(points[:, 0], points[:, 1], points[:,9], c=colors, marker='o')

# # Etichette degli assi
# ax.set_xlabel('Asse X')
# ax.set_ylabel('Asse Y')
# ax.set_zlabel('Asse Z')

# plt.grid(True)
# plt.show()
