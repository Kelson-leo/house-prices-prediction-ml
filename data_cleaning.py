import pandas as pd

def clean_data(df, output_file='cleaned_data.csv'):

    # Removes columns with missing values issues
    cols_to_be_removed = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'LotFrontage',
    'GarageYrBlt', 'MasVnrArea']
    df = df.drop(columns=cols_to_be_removed) # Removido inplace

    # Transforms ordinal columns to numeric
    ordinal_cols = ['FireplaceQu', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
    'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
    for col in ordinal_cols:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("None")
        # Correção do replace (estilo moderno)
        df[col] = df[col].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

    # Fills NA where incorrectly pandas placed NaN
    for c in ['GarageType', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1']:
        df[c] = df[c].fillna('NA')

    # CORREÇÃO AQUI: Voltando os nomes corretos das colunas e valores
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    df['Electrical'] = df['Electrical'].fillna('SBrkr')

    # Saves a copy sem o índice do pandas (deixa o csv mais limpo)
    df.to_csv(output_file, index=False)

    return df

if __name__ == "__main__":
    df = pd.read_csv('raw_data.csv')
    print(f'Original Data: {df.shape}')
    cleaned_df = clean_data(df)

    columns_with_miss = cleaned_df.isna().sum()
    columns_with_miss = columns_with_miss[columns_with_miss!=0]
    print(f'Columns with missing values: {len(columns_with_miss)}')
    if len(columns_with_miss) > 0:
        print(columns_with_miss.sort_values(ascending=False))

    print(f'After Cleaning: {cleaned_df.shape}')