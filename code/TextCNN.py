from keras import Input
from keras.layers import Conv1D, MaxPool1D, Dense, Flatten, concatenate, Embedding,Concatenate,Dropout,SpatialDropout1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks

def textcnn(length, filter_num, kernel_sizes, max_features, embed_size, embedding_matrix):
    #调整输入维度
    inp = Input(shape=(length,))
    #词嵌入
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    #dropout
    x = SpatialDropout1D(0.5)(x)
    
    #分别进行卷积运算，学习特定的文本特征
    conv_0 = Conv1D(filter_num, kernel_size=kernel_sizes[0], kernel_initializer='normal',activation='relu')(x)
    conv_1 = Conv1D(filter_num, kernel_size=kernel_sizes[1], kernel_initializer='normal',activation='relu')(x)
    conv_2 = Conv1D(filter_num, kernel_size=kernel_sizes[2], kernel_initializer='normal',activation='relu')(x)
    conv_3 = Conv1D(filter_num, kernel_size=kernel_sizes[3], kernel_initializer='normal',activation='relu')(x)
    
    #最大池化
    maxpool_0 = MaxPool1D(pool_size=length-kernel_sizes[0]+1)(conv_0)
    maxpool_1 = MaxPool1D(pool_size=length-kernel_sizes[1]+1)(conv_1)
    maxpool_2 = MaxPool1D(pool_size=length-kernel_sizes[2]+1)(conv_2)
    maxpool_3 = MaxPool1D(pool_size=length-kernel_sizes[3]+1)(conv_3)
    
    #将几个通道的结果拼接起来
    z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
    z = Flatten()(z)
    z = Dropout(0.5)(z)
        
    outp = Dense(6, activation="sigmoid")(z)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model