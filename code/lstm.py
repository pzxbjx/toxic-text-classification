from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D,SpatialDropout1D, BatchNormalization
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

def basic_lstm(length, max_features, embed_size, embedding_matrix):
    #调整输入维度
    inp = Input(shape=(length,))
    #词嵌入
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    #dropout
    x = SpatialDropout1D(0.5)(x)
    #双向LSTM
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = BatchNormalization()(x)
    #最大池化
    x = GlobalMaxPool1D()(x)
    #全连接层
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    #sigmoid代替softmax（多标签分类）
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    #损失函数为交叉熵，优化算法为adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model