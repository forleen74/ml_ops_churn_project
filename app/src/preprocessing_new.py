# Import libraries
import pandas as pd
import numpy as np

# это фичи, используемые моделью
col=['регион', 'использование', 'сумма', 'частота_пополнения', 
       'доход', 'сегмент_arpu', 'частота',
       'объем_данных', 'on_net',
       'продукт_1', 'продукт_2', 'секретный_скор', 'pack_freq', 'null_count']


def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file)

    return input_df

# функция препроцессинга
def run_preproc(input_df):

    # Import Train dataset
    train = pd.read_csv('./train_data/train.csv')
    print('Train data imported...')

    # добавление новой фичи - число пропусков в строке данных
    input_df['null_count'] = input_df.isnull().sum(axis=1)

    # здесь пропуски в числовых значениях заменяются медианой
    # категориальные фичи преобразовываются в числовые методом pd.factorize
    output_df=pd.DataFrame()
    for i in col:
        if input_df[i].dtype.name != 'object':
            output_df[i]=input_df[i].copy()
            output_df.loc[output_df[i].isna(), i]=output_df[i].median()
        else:
            output_df[i]=pd.factorize(input_df[i])[0]
            
    return output_df