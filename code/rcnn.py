from keras import Input
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate,GlobalMaxPool1D
from keras.layers import Embedding,Concatenate,Dropout,SpatialDropout1D,GRU,Bidirectional,BatchNormalization
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

def rcnn_model(length, max_features, embed_size, embedding_matrix):
    #调整输入维度
    inp = Input(shape=(length,))
    #词嵌入
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    #dropout
    x_1 = SpatialDropout1D(0.5)(x)
    #双向GRU
    x_2 = Bidirectional(GRU(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x_1)
    #拼接词向量
    x = Concatenate(axis=2)([x,x_2]) 
    x = SpatialDropout1D(0.5)(x)
    #卷积
    x = Conv1D(100, kernel_size=3, kernel_initializer='normal',activation='tanh')(x)
    x = BatchNormalization()(x)
    #最大池化
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    #sigmoid代替softmax（多标签分类）
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    #损失函数为交叉熵，优化算法为adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model