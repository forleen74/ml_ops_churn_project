import pandas as pd

# Import libs to solve classification task
from catboost import CatBoostClassifier

# Make prediction
def make_pred(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = CatBoostClassifier()
    model.load_model('./models/my_catboost_model_new.cbm') #изменил на new

    # трешхолд
    model_th = 0.587

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file)['client_id'],
        'preds': (model.predict_proba(dt)[:, 1] > model_th) * 1
    })
    print('Prediction complete!')

    # Return proba for positive class
    return submission