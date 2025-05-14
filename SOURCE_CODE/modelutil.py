import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential: 
    model = Sequential()

    # 3D Convolutional layers
    model.add(Conv3D(128, kernel_size=3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, kernel_size=3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Flatten time-distributed features
    model.add(TimeDistributed(Flatten()))

    # First BiLSTM layer (Fixed input shape issue)
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))  
    model.add(Dropout(0.5))  # Corrected dropout placement

    # Second BiLSTM layer
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))  
    model.add(Dropout(0.5))

    # Dense layers for classification
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))  

    # Load model weights
    weights_path = r"C:\Users\jeeva\OneDrive\Desktop\LR\.idea\App\lipnet_trained.weights.h5"
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    try:
        model.load_weights(weights_path)
        print("✅ Model weights loaded successfully!")
    except ValueError as e:
        print(f"❌ Error loading weights: {e}")
        print("⚠️ Possible mismatch between model architecture and weights file.")

    return model
