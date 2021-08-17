# Importer les librairies
import numpy as np
import pandas as pd
from tabulate import tabulate

# Importer le dataset
dataset = pd.read_csv('dataset.csv')

#dataset = pd.read_csv('champ.csv')
dataset = dataset.iloc[:, 1:]

# On change les colonnes des noms des équipes par des type Category
dataset["Team 1"] = dataset["Team 1"].astype("category")
dataset["Team 2"] = dataset["Team 2"].astype("category")

dataset["Team 1 Coach"] = dataset["Team 1 Coach"].astype("category")
dataset["Team 2 Coach"] = dataset["Team 2 Coach"].astype("category")

dataset["Team 1 Formation"] = dataset["Team 1 Formation"].astype("category")
dataset["Team 2 Formation"] = dataset["Team 2 Formation"].astype("category")

# Transformation des équipes en numéro
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
teams = pd.concat([dataset["Team 1"], dataset["Team 2"]])
teams = teams.drop_duplicates()
teams = teams.sort_values(ignore_index=True)
encoder.fit(teams)
dataset["Team 1"] = encoder.transform(dataset["Team 1"])
dataset["Team 2"] = encoder.transform(dataset["Team 2"])

# Transformation des coach en numéro
encoder_coach = LabelEncoder()
coachs = pd.concat([dataset["Team 1 Coach"], dataset["Team 2 Coach"]])
coachs = coachs.drop_duplicates()
coachs = coachs.sort_values(ignore_index=True)
encoder_coach.fit(coachs)
dataset["Team 1 Coach"] = encoder_coach.transform(dataset["Team 1 Coach"])
dataset["Team 2 Coach"] = encoder_coach.transform(dataset["Team 2 Coach"])

# Transformation des formations en numéro
encoder_tactics = LabelEncoder()
tactics = pd.concat([dataset["Team 1 Formation"], dataset["Team 2 Formation"]])
tactics = tactics.drop_duplicates()
tactics = tactics.sort_values(ignore_index=True)
encoder_tactics.fit(tactics)
dataset["Team 1 Formation"] = encoder_tactics.transform(dataset["Team 1 Formation"])
dataset["Team 2 Formation"] = encoder_tactics.transform(dataset["Team 2 Formation"])

# Diviser le dataset entre le Training set et le Test set
from sklearn.model_selection import train_test_split
x = dataset[
    ['Date',
     'Team 1',
     'Team 1 Coach',
     'Team 1 Formation',
     'Team 1 Value',
     'Team 2',
     'Team 2 Coach',
     'Team 2 Formation',
     'Team 2 Value']
    ]
y = dataset[
    ['Team 1 Goals',
     'Team 1 Tirs cadres',
     'Team 1 Tirs non cadres',
     'Team 1 Possession',
     'Team 1 Passes',
     'Team 1 Precision passes',
     'Team 1 Fautes',
     'Team 1 Cartons jaunes',
     'Team 1 Cartons rouges',
     'Team 1 HJ',
     'Team 1 Corners',
     'Team 2 Goals',
     'Team 2 Tirs cadres',
     'Team 2 Tirs non cadres',
     'Team 2 Possession',
     'Team 2 Passes',
     'Team 2 Precision passes',
     'Team 2 Fautes',
     'Team 2 Cartons jaunes',
     'Team 2 Cartons rouges',
     'Team 2 HJ',
     'Team 2 Corners']
    ]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# On utilise un modèle
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion="mse")
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)
y_predict = np.round(y_predict)

italie = {"Team": [11],
          "Coach": [12],
          "Formation": [8],
          "Value": [411500000]}

angleterre = {"Team": [1],
              "Coach": [21],
              "Formation": [7],
              "Value": [605000000]}

def PredictScore(equipe1, equipe2):
    match = pd.DataFrame({"Date": [2021],
                     "Team 1": equipe1["Team"],
                     "Team 1 Coach": equipe1["Coach"],
                     "Team 1 Formation": equipe1["Formation"],
                     "Team 1 Value": equipe1["Value"],
                     "Team 2": equipe2["Team"],
                     "Team 2 Coach": equipe2["Coach"],
                     "Team 2 Formation": equipe2["Formation"],
                     "Team 2 Value": equipe2["Value"]})
    match = sc.transform(match)
    return np.round(regressor.predict(match))

def PrintScore(score):
    print(tabulate(
    [
     [score[0, 0], "Score", score[0, 11]],
     [(score[0, 1] + score[0, 2]), "Shoots", (score[0, 12] + score[0, 13])],
     [score[0, 1], "On goal", score[0, 12]],
     [score[0, 3], "Possession", score[0, 14]],
     [score[0, 4], "Passes", score[0, 15]],
     [score[0, 5], "Accuracy", score[0, 16]],
     [score[0, 6], "Fools", score[0, 17]],
     [score[0, 7], "Yellow cards", score[0, 18]],
     [score[0, 8], "Red cards", score[0, 19]],
     [score[0, 9], "Offside", score[0, 20]],
     [score[0, 10], "Corners", score[0, 21]],
    ],
    headers=['Italy', "-", 'England'],
    colalign=("right","center","left")))

match_score = PredictScore(italie, angleterre)
PrintScore(match_score)
