import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings("ignore")

pd.set_option('display.max_columns', None)


r = requests.get('*************')
soup = BeautifulSoup(r.content, "lxml")
games = soup.find_all("tr", attrs={"data-ish": "1"})
games45 = soup.find_all("tr", attrs={"data-ish": "0"})

list1 = []
for i in games:
    list1.append(i.find("td", attrs={"class": "text-center td_league"}).find("div", attrs={"href": ""}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_status"}).find("span", attrs={
        "class": "match_status_minutes"}).text)
    list1.append(i.find("td", attrs={"class": "text-right match_home"}).find("span", attrs={"class": ""}).text)
    list1.append(i.find("td", attrs={"class": "text-left match_away"}).find("span", attrs={"class": ""}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_goal"}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_corner"}).find("span", attrs={
        "class": "span_match_corner"}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_corner"}).find("span", attrs={
        "class": "span_half_corner"}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_attach"}).find("div", attrs={
        "class": "match_dangerous_attacks_div"}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_attach"}).find("div", attrs={
        "class": "match_dangerous_attacks_half_div"}).text)
    list1.append(
        i.find("td", attrs={"class": "text-center match_shoot"}).find("div", attrs={"class": "match_shoot_div"}).text)
    list1.append(i.find("td", attrs={"class": "text-center match_shoot"}).find("div", attrs={
        "class": "match_shoot_half_div"}).text)

list2 = []
for i in games45:
    list2.append(i.find("td", attrs={"class": "text-center td_league"}).find("div", attrs={"href": ""}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_status"}).find("span", attrs={
        "class": "match_status_minutes"}).text)
    list2.append(i.find("td", attrs={"class": "text-right match_home"}).find("span", attrs={"class": ""}).text)
    list2.append(i.find("td", attrs={"class": "text-left match_away"}).find("span", attrs={"class": ""}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_goal"}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_corner"}).find("span", attrs={
        "class": "span_match_corner"}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_corner"}).find("span", attrs={
        "class": "span_half_corner"}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_attach"}).find("div", attrs={
        "class": "match_dangerous_attacks_div"}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_attach"}).find("div", attrs={
        "class": "match_dangerous_attacks_half_div"}).text)
    list2.append(
        i.find("td", attrs={"class": "text-center match_shoot"}).find("div", attrs={"class": "match_shoot_div"}).text)
    list2.append(i.find("td", attrs={"class": "text-center match_shoot"}).find("div", attrs={
        "class": "match_shoot_half_div"}).text)

column_names = ["LEAGUE", "MINUTES", "TEAM_1_NAME", "TEAM_2_NAME", "SCORE", "TOTAL_CORNER", "HALFTIME_CORNER",
                "TOTAL_ATTACKS", "HALFTIME_ATTACKS", "TOTAL_SHOTS", "HALFTIME_SHOTS"]
df = pd.DataFrame(list1)
df2 = pd.DataFrame(list2)
df.columns = ["temp"]
df2.columns = ["temp"]
df = pd.DataFrame(df.groupby(df.index // 11)["temp"].apply(list).values.tolist())
df2 = pd.DataFrame(df2.groupby(df2.index // 11)["temp"].apply(list).values.tolist())
df.columns = column_names
df2.columns = column_names

df2['MINUTES'] = df2['MINUTES'].replace("", value=0).astype("int")

list3 = []
for index, row in df2.iterrows():
    if row['MINUTES'] > 0:
        list3.append(row)
df3 = pd.DataFrame(list3, columns=column_names)

df_final = pd.concat([df, df3], ignore_index=True)

def data_col_pre(dataframe):
    dataframe = dataframe.replace("", "0-0")
    dataframe = dataframe.replace(" ", "0-0")
    dataframe.replace(np.nan, 0, inplace=True)

    dataframe["LEAGUE"] = dataframe["LEAGUE"].str.replace('\n', '')
    dataframe['HALFTIME_CORNER'] = dataframe['HALFTIME_CORNER'].str.replace('(', '').str.replace(')', '')
    dataframe[['HALFTIME_CORNER_TEAM_1', 'HALFTIME_CORNER_TEAM_2']] = dataframe["HALFTIME_CORNER"].str.split("-",
                                                                                                             expand=True)
    dataframe[['SCORE_TEAM_1', 'SCORE_TEAM_2']] = dataframe["SCORE"].str.split("-", expand=True)
    dataframe[['TOTAL_CORNER_TEAM_1', 'TOTAL_CORNER_TEAM_2']] = dataframe["TOTAL_CORNER"].str.split("-", expand=True)
    dataframe[['TOTAL_ATTACKS_TEAM_1', 'TOTAL_ATTACKS_TEAM_2']] = dataframe["TOTAL_ATTACKS"].str.split("-", expand=True)
    dataframe[['HALFTIME_ATTACKS_TEAM_1', 'HALFTIME_ATTACKS_TEAM_2']] = dataframe["HALFTIME_ATTACKS"].str.split("-",
                                                                                                                expand=True)
    dataframe[['TOTAL_SHOTS_TEAM_1', 'TOTAL_SHOTS_TEAM_2']] = dataframe["TOTAL_SHOTS"].str.split("-", expand=True)
    dataframe[['HALFTIME_SHOTS_TEAM_1', 'HALFTIME_SHOTS_TEAM_2']] = dataframe["HALFTIME_SHOTS"].str.split("-",
                                                                                                          expand=True)
    dataframe.replace(np.nan, 0, inplace=True)
    dataframe = dataframe.replace(" ", value=0)
    dataframe = dataframe.replace("Half", value=45)
    dataframe = dataframe.replace("", value=0)

    drop_list = ["SCORE", "TOTAL_CORNER", "HALFTIME_CORNER", "TOTAL_ATTACKS", "HALFTIME_ATTACKS", "TOTAL_SHOTS",
                 "HALFTIME_SHOTS"]
    dataframe.drop(drop_list, axis=1, inplace=True)

    non_int_list = ["LEAGUE", "TEAM_1_NAME", "TEAM_2_NAME"]
    for i in [col for col in dataframe.columns if col not in non_int_list]:
        dataframe[i] = dataframe[i].astype("int")

    dataframe = dataframe[
        ["LEAGUE", "TEAM_1_NAME", "SCORE_TEAM_1", "SCORE_TEAM_2", "TEAM_2_NAME", "TOTAL_CORNER_TEAM_1",
         "TOTAL_CORNER_TEAM_2",
         "TOTAL_ATTACKS_TEAM_1", "TOTAL_ATTACKS_TEAM_2", "TOTAL_SHOTS_TEAM_1", "TOTAL_SHOTS_TEAM_2",
         "HALFTIME_CORNER_TEAM_1",
         "HALFTIME_CORNER_TEAM_2", "HALFTIME_ATTACKS_TEAM_1", "HALFTIME_ATTACKS_TEAM_2",
         "HALFTIME_SHOTS_TEAM_1", "HALFTIME_SHOTS_TEAM_2"]]
    dataframe["TOTAL_SCORE"] = dataframe["SCORE_TEAM_1"] + dataframe["SCORE_TEAM_2"]
    return dataframe

df_final = data_col_pre(df_final)

old_data = pd.read_csv("scoredata.csv", error_bad_lines=False)

with open("scoredata.csv", mode="a", newline='' ,encoding='utf-8') as file:

    df_final_csv = df_final.to_csv(index=False, sep=",", header=False)
    file.write(df_final_csv)

for_x_drop_list  = ["LEAGUE","TOTAL_SCORE","TEAM_1_NAME","TEAM_2_NAME","SCORE_TEAM_1","SCORE_TEAM_2"]

X = old_data.drop(for_x_drop_list, axis=1)
y = old_data["TOTAL_SCORE"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

lgbm_model = LGBMRegressor(random_state=123).fit(X_train, y_train)

y_pred_train = lgbm_model.predict(X_train)

y_pred_test = lgbm_model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

cv_score = np.sqrt(-cross_val_score(estimator=lgbm_model, X=X, y=y, cv=10, scoring="neg_mean_squared_error"))

print(f"Train Set RMSE: {rmse_train}, Test Set RMSE: {rmse_test} ")
print(f"CV Score Mean: {cv_score.mean()}, Standard Deviation: {cv_score.std()}")

iterations = 5
lr = []
sf = []
nl = []
md = []

for i in range(iterations):
    lr.append(np.random.uniform(0, 1))
    sf.append(np.random.uniform(0, 1))
    nl.append(np.random.randint(20, 300))
    md.append(np.random.randint(5, 200))

lgbm_grid = {"learning_rate":lr,"boosting_type":["gbdt","dart","goss"],"objective":["regression"],
             "metric":["rmse"],"sub_feature":sf,"num_leaves":nl,"max_depth": md }

lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(param_grid=lgbm_grid,estimator=lgbm,cv=5, n_jobs=-1, verbose=-1).fit(X,y)

lgbm_cv_model.best_params_

lgbm_final_model = lgbm.set_params(**lgbm_cv_model.best_params_).fit(X,y)
cv_score = np.sqrt(-cross_val_score(estimator=lgbm_final_model, X=X, y=y, cv=10, scoring="neg_mean_squared_error", n_jobs=-1,verbose=-1))


cv_score.mean()


def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()


plot_importance(lgbm_final_model, X)


X_train = old_data.drop(for_x_drop_list, axis=1)
X_test = df_final.drop(for_x_drop_list, axis=1)

y_train = old_data["TOTAL_SCORE"]
y_test = df_final["TOTAL_SCORE"]

lgbm_live_model = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_live_model.predict(X_test)

expected = pd.DataFrame(y_pred, columns=["EXP_SCORE"])
expected["EXP_SCORE"] = expected["EXP_SCORE"].apply(lambda x: 0 if x<0 else x)
df_final["EXPECTED_TOTAL_SCORE"] = expected["EXP_SCORE"]

df_final