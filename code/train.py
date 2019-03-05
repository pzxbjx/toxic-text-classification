from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint, Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#计算AUC分数
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
def modeltrainer(model, X, y, batch_size, epoch, file_path):
    
    #训练集，验证集的划分
    X_tra, X_val, y_tra, y_val = train_test_split(X, y, train_size=0.9, random_state=233)
    #根据验证集的损失函数来保存最佳的模型参数
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, save_weights_only=True, mode = "min")
    ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
    #如果连续5个epoch模型精度没有提高，认定模型出现了过拟合的现象，终止训练
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
    model.fit(X_tra, y_tra, batch_size, epoch, validation_data=(X_val, y_val),verbose = 1, callbacks = [ra_val, check_point, early_stop])
    return model