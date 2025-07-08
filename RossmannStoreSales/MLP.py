import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# process data
def process_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsWeekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Day_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['Day_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)

    df = df.drop(columns=['Date','DayOfYear'])
    return df

def process_competition_features(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)
    df['CompetitionOpenDurationMonths'] = df.apply(
        lambda row: (row['Year'] - row['CompetitionOpenSinceYear']) * 12 +
                    (row['Month'] - row['CompetitionOpenSinceMonth'])
        if row['CompetitionOpenSinceYear'] > 0 and row['CompetitionOpenSinceMonth'] > 0 else 0,
        axis=1
    )
    df = df.drop(columns=['Year', 'Month'])
    return df

def process_promo_interval(df):
    df['PromoInterval'] = df['PromoInterval'].fillna('')
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in months:
        df[f'IsPromo{month}'] = df['PromoInterval'].apply(lambda x: int(month in x.split(',')))
    df = df.drop(columns=['PromoInterval'])
    return df

def process_promo2_since_year(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)
    df['Promo2YearsActive'] = df.apply(
        lambda row: row['Year'] - row['Promo2SinceYear']
        if row['Promo2SinceYear'] > 0 else 0,
        axis=1
    )
    df = df.drop(columns=['Year'])
    return df

train_combined = pd.read_csv('combined_training_data.csv')
train_combined = process_promo_interval(train_combined)
train_combined = process_promo2_since_year(train_combined)
train_combined = process_date(train_combined)

X = train_combined.drop(columns=['Sales', 'Customers'])
X['StateHoliday'] = X['StateHoliday'].astype(str)
X = pd.get_dummies(
    X,
    columns=['Assortment', 'StoreType', 'StateHoliday', 'DayOfWeek'],
    drop_first=True
)
X = X.astype(float)
y = train_combined['Sales']

X['CompetitionDistance'] = X['CompetitionDistance'].fillna(-1)
X = X.fillna(0)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
columns_to_scale = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                    'Promo2SinceWeek', 'Promo2YearsActive', 'Year', 'Month', 'Day',
                    'WeekOfYear']
X_train_scaled = X_train.copy()
X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_val_scaled = X_validation.copy()
X_val_scaled[columns_to_scale] = scaler.transform(X_validation[columns_to_scale])
X_val_scaled.to_csv('X_val_scaled.csv', index=False)

# Modelling
def build_mlp_model(input_dim):
    model = Sequential([
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        # layers.Dense(256, activation='relu'),
        # layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='linear')
    ])
    return model

OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-4)
mlp_model = build_mlp_model(X_train_scaled.shape[1])
mlp_model.compile(optimizer=OPTIMIZER, loss='mean_squared_error', metrics=['mae','mse'])

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)
lr_reduce = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6, verbose=1
)

history = mlp_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_validation),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

mlp_model.save('rossmann_mlp_model.h5')
print("MLP已建模")


plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('MLP Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('mlp_loss_curve.png', dpi=300)
plt.show()


# Predicting
test_combined = pd.read_csv('combined_test_data.csv')
test_combined = process_promo_interval(test_combined)
test_combined = process_promo2_since_year(test_combined)
test_combined = process_date(test_combined)

X_test = test_combined.drop(columns=['Id'])
X_test['StateHoliday'] = X_test['StateHoliday'].astype(str)
X_test = pd.get_dummies(
    X_test,
    columns=['Assortment', 'StoreType', 'StateHoliday', 'DayOfWeek'],
    drop_first=True
)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test['CompetitionDistance'] = X_test['CompetitionDistance'].fillna(-1)
X_test = X_test.fillna(0)

X_test_scaled = X_test.copy()
X_test_scaled = X_test_scaled.astype(float)

# X_test_scaled.to_csv('X_test_before_scaled.csv', index=False) # 确认哪些列需要scale
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
X_test_scaled.to_csv('X_test_scaled.csv', index=False)

predictions = mlp_model.predict(X_test_scaled)

predictions_df = pd.DataFrame({
    'Id': test_combined['Id'],
    'Sales': predictions.flatten().clip(0)  # 防止负值
})
predictions_df.to_csv('rossmann_mlp_predictions.csv', index=False)
print('Complete')
