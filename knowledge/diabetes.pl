'Outcome'(Age, BMI, BloodPressure, DiabetesPedigreeFunction, Glucose, Insulin, Pregnancies, SkinThickness, 1) :-
    Glucose > 126, BMI > 30.
'Outcome'(Age, BMI, BloodPressure, DiabetesPedigreeFunction, Glucose, Insulin, Pregnancies, SkinThickness, 0) :-
    Glucose =< 100, BMI =< 25.
