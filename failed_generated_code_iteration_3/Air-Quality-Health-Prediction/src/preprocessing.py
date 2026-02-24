import pandas as pd
import numpy as np

def load_and_preprocess(filepath):

    df = pd.read_csv(filepath, sep=None, encoding='utf-8-sig', engine='python', decimal=',')


    df.columns = df.columns.str.strip()
    if '﻿Date' in df.columns:
        df.rename(columns={'﻿Date': 'Date'}, inplace=True)


    print("\n📊 Raw Data Sample:")
    print(df.head(5))

    df.dropna(axis=1, how='all', inplace=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


    df['Date'] = df['Date'].astype(str).str.strip()
    df['Time'] = df['Time'].astype(str).str.strip().str.replace('.', ':', regex=False)

    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        errors='coerce',
        dayfirst=True
    )


    print("\n📆 Parsed Datetime Sample:")
    print(df['Datetime'].head())
    print("\n✅ Parsed datetime count:", df['Datetime'].notna().sum())


    df = df[df['Datetime'].notna()]
    df.set_index('Datetime', inplace=True)


    pollutants = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)', 'T', 'RH']
    df[pollutants] = df[pollutants].apply(pd.to_numeric, errors='coerce')


    daily_df = df[pollutants].resample('D').mean().dropna()


    np.random.seed(42)
    daily_df['Hospital_Visits'] = (
        daily_df['CO(GT)'] * 10 +
        daily_df['NOx(GT)'] * 0.2 +
        daily_df['NO2(GT)'] * 0.5 +
        np.random.normal(0, 5, len(daily_df))
    ).astype(int)


    threshold = daily_df['Hospital_Visits'].mean() + daily_df['Hospital_Visits'].std()
    daily_df['Risk_Label'] = (daily_df['Hospital_Visits'] > threshold).astype(int)


    print("\n✅ Final processed data shape:", daily_df.shape)
    print(daily_df.head())

    return daily_df