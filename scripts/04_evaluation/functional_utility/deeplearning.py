from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization
from tensorflow.keras.layers import Activation, AveragePooling2D, Dropout, Flatten, Dense

def EEGNet(nb_classes, Chans=32, Samples=500, dropoutRate=0.5):

    input1 = Input(shape=(Chans, Samples, 1))

    # Temporal convolution
    block1 = Conv2D(16, (1, 64), padding='same', use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    # Spatial filtering (like ICA / CSP)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2)(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    # Separable convolution (feature refinement)
    block2 = SeparableConv2D(32, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes, activation='softmax')(flatten)

    return Model(inputs=input1, outputs=dense)

model = EEGNet(nb_classes=2, Chans=X.shape[1], Samples=X.shape[2])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

auc_scores = []

for train_idx, test_idx in skf.split(X, y):

    model = EEGNet(...)

    model.fit(X[train_idx], y[train_idx])

    preds = model.predict(X[test_idx])[:,1]

    auc = roc_auc_score(y[test_idx], preds)
    auc_scores.append(auc)