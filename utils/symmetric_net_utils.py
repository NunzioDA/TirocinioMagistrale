
import numpy as np

def create_4_tuple(_X,_y, knowledge_mask):
    """
    This function creates three 4-tuples based on the given dataset and the knowledge mask.

    Args:
        _X : dataset
        _y : target
        knowledge_mask : boolean mask that identifies knowledge elements.

    Returns:
        3-tuple containing:
        - A 4-tuple with:
        . Elements of the dataset that are NOT part of the knowledge and have target 0
        . Elements of the dataset that are NOT part of the knowledge and have target 1
        . Elements of the dataset that are part of the knowledge and have target 0
        . Elements of the dataset that are part of the knowledge and have target 1

        - A 4-tuple with:
        . The target for each element of the previous 4-tuple
        ##  Note: this is for debugging purposes only! This will be removed later, as
            the target is always the same -> [0, 1, 0, 1] 

        - A 4-tuple with the weights for each element in the first 4-tuple (currently set to 1)
    """

    knowledge_X = _X[knowledge_mask].copy()
    knowledge_X["Outcome"] = _y[knowledge_mask].copy()

    knowledge_X_0 = knowledge_X[knowledge_X["Outcome"] == 0].copy()
    knowledge_X_1 = knowledge_X[knowledge_X["Outcome"] == 1].copy()

    _x_0 = _X[(knowledge_mask == False)].copy()
    _y_0 = _y[(knowledge_mask == False)].copy()
    # [(knowledge_mask == False)]
    # [(knowledge_mask == False)]

    
    _x_0_0 = _x_0[_y_0==0].copy()
    _x_0_1 = _x_0[_y_0==1].copy()

    max_shape = max(_x_0_0.shape[0],_x_0_1.shape[0])

    _x_0_1 = _x_0_1.sample(n=max_shape,replace=True, random_state=42)
    _x_0_0 = _x_0_0.sample(n=max_shape,replace=True, random_state=42)

    _y_0_0 = np.zeros(_x_0_0.shape[0])
    _y_0_1 = np.ones(_x_0_1.shape[0])

    _x_1 = knowledge_X_0.sample(n=max_shape,replace=True, random_state=42)
    _y_1 = _x_1["Outcome"]
    _x_1.drop(columns=["Outcome"],inplace=True)

    _x_2 = knowledge_X_1.sample(n=max_shape,replace=True, random_state=42)
    _y_2 = _x_2["Outcome"]
    _x_2.drop(columns=["Outcome"],inplace=True)

    class_sample_weight_0_0 = np.ones_like(_y_0_0)
    class_sample_weight_0_1 = np.ones_like(_y_0_0)
    class_sample_weight_1 = np.ones_like(_y_1)
    class_sample_weight_2 = np.ones_like(_y_2)

    _return_x = [_x_0_0,_x_0_1,_x_1,_x_2]
    _return_y = np.column_stack([_y_0_0,_y_0_1,_y_1,_y_2])
    _return_sample_weight = np.column_stack([class_sample_weight_0_0, class_sample_weight_0_1, class_sample_weight_1,class_sample_weight_2])


    return (_return_x, _return_y, _return_sample_weight)

# 

def get_mask(_X, _y, maxes):
    """
        This function creates the knowledge mask.

        Args:
            _X: dataset
            _y: target
            maxes: Required if the dataset is normalized; used to compare with unnormalized values
    """
    mask1 = ((_X["Glucose"] > 126 / maxes[1]) & (_X["BMI"] > 30 / maxes[5]) & (_y == 1))
    mask2 = ((_X["Glucose"] <= 100 / maxes[1]) & (_X["BMI"] <= 25 / maxes[5]) & (_y == 0))
    # mask3 = ((_X["Glucose"] <= 121 / maxes[1]) & (_X["BMI"] > 40 / maxes[5]) & (_y == 1))
    # mask4 = ((_X["Glucose"] > 121 / maxes[1]) & (_X["BMI"] <= 29 / maxes[5]) & (_X["Age"] > 30 / maxes[7]) & (_y == 1))
    # mask5 = ((_X["Glucose"] > 121 / maxes[1]) & (_X["BMI"] > 29 / maxes[5]) & (_y == 1))
    # mask6 = ((_X["Glucose"] <= 121 / maxes[1]) & (_X["BMI"] <= 40 / maxes[5]) & 
    #          (_X["DiabetesPedigreeFunction"] > 0) & (_X["Age"] > 40 / maxes[7]) & (_y == 1))

    mask = mask1 | mask2 #| mask3 | mask4 | mask5 | mask6
    return mask