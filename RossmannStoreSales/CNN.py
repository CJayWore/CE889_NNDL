import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras import Sequential
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def process_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['IsWeekend'] = (df['Date'].dt.weekday >= 5).astype(int)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfYear'] =df['Date'].dt.dayofyear
    df['Day_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['Day_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df = df.drop(columns=['Date'])
    return df

def process_competition_features(df):
    # 确保 'Date' 列是 datetime 格式
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0).astype(int)
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0).astype(int)

    # 计算开店持续时间（以月为单位）
    df['CompetitionOpenDurationMonths'] = df.apply(
        lambda row: (row['Year'] - row['CompetitionOpenSinceYear']) * 12 +
                    (row['Month'] - row['CompetitionOpenSinceMonth'])
        if row['CompetitionOpenSinceYear'] > 0 and row['CompetitionOpenSinceMonth'] > 0 else 0,
        axis=1
    )
    df = df.drop(columns=['Year', 'Month'])
    return df

def process_promo_interval(df):
    # Fill null values with an empty string
    df['PromoInterval'] = df['PromoInterval'].fillna('')

    # Create new binary features for each month
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in months:
        df[f'IsPromo{month}'] = df['PromoInterval'].apply(lambda x: int(month in x.split(',')))
    df = df.drop(columns=['PromoInterval'])
    return df

def process_promo2_since_year(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    # 填充 'Promo2SinceYear' 的缺失值，并确保类型为整数
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0).astype(int)

    # 计算促销活动的持续年份差
    df['Promo2YearsActive'] = df.apply(
        lambda row: row['Year'] - row['Promo2SinceYear']
        if row['Promo2SinceYear'] > 0 else 0,
        axis=1
    )
    df = df.drop(columns=['Year'])
    return df

# Load processed data
train_combined = pd.read_csv('combined_training_data.csv')

train_combined = process_promo_interval(train_combined)
train_combined = process_promo2_since_year(train_combined)
train_combined = process_date(train_combined)


# Prepare data for training
X = train_combined.drop(columns=['Sales', 'Customers'])

X['StateHoliday'] = X['StateHoliday'].astype(str)
X = pd.get_dummies(
    X,
    columns=['Assortment', 'StoreType', 'StateHoliday', 'DayOfWeek'],
    drop_first=True  # Avoid multicollinearity
)
# print(X.dtypes)
X = X.astype(float)
# print(X.dtypes)
y = train_combined['Sales']

# Fill missing values in X (e.g., CompetitionDistance and Promo2SinceYear)
X = X.fillna(0)
X.to_csv('X_before_scaler.csv', index=False)
# Split the data
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
columns_to_scale = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                    'Promo2SinceWeek', 'Promo2YearsActive', 'Year', 'Month', 'Day',
                    'WeekOfYear', 'DayOfYear']
columns_not_scale = [col for col in X_train.columns if col not in columns_to_scale]
X_train_scaled = X_train.copy()
X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_val_scaled = X_validation.copy()
X_val_scaled[columns_to_scale] = scaler.transform(X_validation[columns_to_scale])
# print("Scaled training data:")
# print(X_train_scaled.head())
X_train_scaled.to_csv('X_train_scaled.csv', index=False)

# Reshape data for CNN input (2D convolution)
X_train_cnn = X_train_scaled.values.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1], 1))
X_validation_cnn = X_val_scaled.values.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1], 1))


# Load test data
test_combined = pd.read_csv('combined_test_data.csv')

test_combined = process_promo_interval(test_combined)
test_combined = process_promo2_since_year(test_combined)
test_combined = process_date(test_combined)

X_test = test_combined.drop(columns=['Id'])
X_test['StateHoliday'] = X_test['StateHoliday'].astype(str)
X_test = pd.get_dummies(
    X_test,
    columns=['Assortment', 'StoreType', 'StateHoliday', 'DayOfWeek'],
    drop_first=True  # Match training preprocessing
)

# Ensure test data has the same columns as train data
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test.to_csv('X_test_before_scale.csv', index=False)

X_test_scaled = X_test.copy()
X_test_scaled = X_test_scaled.astype(float)
X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
X_test_scaled.to_csv('X_test_after_scale.csv', index=False)

X_test_cnn = X_test_scaled.to_numpy().reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1], 1))


# Build 2D CNN model
model = Sequential([
    # Conv Layer 1
    layers.Conv2D(64, kernel_size=(1, 3), activation='relu', input_shape=(1, X_train_scaled.shape[1], 1)),
    layers.MaxPooling2D(pool_size=(1, 2)),
    layers.Dropout(0.2),

    # Conv Layer 2
    layers.Conv2D(128, kernel_size=(1, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(1, 2)),
    layers.Dropout(0.2),

    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='linear')  # Regression output
])

# Optimizer and compilation
OPTIMIZER = keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=OPTIMIZER, loss='mean_squared_error', metrics=['mae'])

# Callbacks
lr_reduce = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
)

# Train the model
history = model.fit(
    X_train_cnn, y_train,
    validation_data=(X_validation_cnn, y_validation),
    epochs=400,
    batch_size=64,
    callbacks=[lr_reduce, early_stop],
    verbose=1
)

# Save the model
model.save('rossmann_cnn_model.h5')
print("Model has been saved")

# Plot training & validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig('loss_curve.png', dpi=300)
plt.show()


# Predict on test data
model = keras.models.load_model('rossmann_cnn_model.h5')
predictions = model.predict(X_test_cnn)

# Save predictions
predictions_df = pd.DataFrame({
    'Id': test_combined['Id'],
    'Sales': predictions.flatten().clip(0)
})
predictions_df.to_csv('rossmann_cnn_test_predictions.csv', index=False)



