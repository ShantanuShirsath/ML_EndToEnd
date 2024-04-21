# Sleep Disorder Prediction

The following project is a demonstration of a Machine learning project with modular programming and use of preprocessing, training and prediction pipeline to derive the results.

## Project Details

**Exploratory Data analysis**: Exploratory data analysis is done to get the replationship between sleep disorder and other factors affecting the sleep. The analysis reveals interesting facts about factors that affect the sleep and effect of stress, occupation, Gender etc on sleep cycle.

**Data Preprocessing**: A preprocessing pipeline is used to clean the data which include,
- encoding of categorical features
- encoding of labels
- imputing  missing values in numerical features
- Scaling numerical values


**Model Training**: Different classification models which include Ada-Boost, K-Nearest Neighbour, Decision tree, Random Forest, XGBoost, CatBoost are used to train on data and give out results. 

**Prediction pipeline**: A prediction pipeline is deployed using a flask Web app wherein the data is collected from user and passed through the predict pipeline to get the result.
The result are displayed on the webpage
