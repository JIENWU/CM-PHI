import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Lambda, GaussianNoise, \
    BatchNormalization, Reshape, dot, Activation, concatenate, AveragePooling1D, GlobalAveragePooling1D
from keras.engine.topology import Layer
from keras.utils import plot_model
from keras.datasets import mnist
from keras import backend as K
from random import shuffle
from keras.callbacks import ReduceLROnPlateau
import csv

csv.field_size_limit(500 * 1024 * 1024)
# 读的字段太大，https://blog.csdn.net/dm_learner/article/details/79028357
# import sys   # 或者
# import csv
# csv.field_size_limit(sys.maxsize)
import numpy as np
import math


# 定义函数
def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return


def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        for i in range(len(row)):  # 转换数据类型
            row[i] = float(row[i])
        SaveList.append(row)
    return


def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 1
        while counter < len(row):
            row[counter] = float(row[counter])
            counter = counter + 1
        SaveList.append(row)
    return


def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


def GenerateEmbeddingFeature(SequenceList, EmbeddingList, PaddingLength):  # 产生所有miRNA/drug的embedding表示
    SampleFeature = []

    counter = 0
    while counter < len(SequenceList):
        PairFeature = []
        PairFeature.append(SequenceList[counter][0])  # 加入名称

        FeatureMatrix = []
        counter1 = 0  # 生成特征矩阵
        while counter1 < PaddingLength:  # 截取长度
            row = []
            counter2 = 0
            while counter2 < len(EmbeddingList[0]) - 1:  # embedding长度
                row.append(0)
                counter2 = counter2 + 1
            FeatureMatrix.append(row)
            counter1 = counter1 + 1

        try:
            counter3 = 0
            while counter3 < PaddingLength:
                counter4 = 0
                while counter4 < len(EmbeddingList):
                    if SequenceList[counter][1][counter3] == EmbeddingList[counter4][0]:
                        FeatureMatrix[counter3] = EmbeddingList[counter4][1:]
                        break
                    counter4 = counter4 + 1
                counter3 = counter3 + 1
        except:
            pass

        PairFeature.append(FeatureMatrix)
        SampleFeature.append(PairFeature)
        counter = counter + 1
    return SampleFeature


def GenerateSampleFeature(InteractionList, EmbeddingFeature1, EmbeddingFeature2):
    SampleFeature1 = []
    SampleFeature2 = []

    counter = 0
    while counter < len(InteractionList):
        Pair1 = InteractionList[counter][0]
        Pair2 = InteractionList[counter][1]

        counter1 = 0
        while counter1 < len(EmbeddingFeature1):
            if EmbeddingFeature1[counter1][0] == Pair1:
                SampleFeature1.append(EmbeddingFeature1[counter1][1])
                break
            counter1 = counter1 + 1

        counter2 = 0
        while counter2 < len(EmbeddingFeature2):
            if EmbeddingFeature2[counter2][0] == Pair2:
                SampleFeature2.append(EmbeddingFeature2[counter2][1])
                break
            counter2 = counter2 + 1

        counter = counter + 1
    SampleFeature1, SampleFeature2 = np.array(SampleFeature1), np.array(SampleFeature2)
    SampleFeature1.astype('float64')
    SampleFeature2.astype('float64')
    SampleFeature1 = SampleFeature1.reshape(SampleFeature1.shape[0], SampleFeature1.shape[1], SampleFeature1.shape[2],
                                            1)
    SampleFeature2 = SampleFeature2.reshape(SampleFeature2.shape[0], SampleFeature2.shape[1], SampleFeature2.shape[2],
                                            1)
    return SampleFeature1, SampleFeature2


def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]

        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append(NodeBehavior[m][1:])
                break

        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append(NodeBehavior[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float64')
    SampleFeature2 = np.array(SampleFeature2).astype('float64')

    return SampleFeature1, SampleFeature2


def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label


def scheduler(epoch):
    # 每隔100个epoch，学习率减小为原来的1/2
    if epoch % 4 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.01)
        print("lr changed to {}".format(lr * 0.01))
    return K.get_value(model.optimizer.lr)


def MyChange(list):
    List = []
    for i in range(len(list)):
        row = []
        row.append(i)  # x
        row.append(float(list[i]))  # y
        List.append(row)
    return List


if __name__ == '__main__':

    # ---------特征输入-------

    AllNodeBehavior1 = []
    ReadMyCsv1(AllNodeBehavior1, 'topological_features.csv')

    AllNodeBehavior2 = []
    ReadMyCsv1(AllNodeBehavior2, '2. GateCNprotein_features.csv')

    # ---------划分训练-------

    PositiveSample_Train = []
    ReadMyCsv1(PositiveSample_Train, 'PositiveSample_Train_PBI.csv')
    PositiveSample_Validation = []
    ReadMyCsv1(PositiveSample_Validation, 'PositiveSample_Validation_PBI.csv')
    PositiveSample_Test = []
    ReadMyCsv1(PositiveSample_Test, 'PositiveSample_Test_PBI.csv')

    NegativeSample_Train = []
    ReadMyCsv1(NegativeSample_Train, 'NegativeSample_Train_PBI.csv')
    NegativeSample_Validation = []
    ReadMyCsv1(NegativeSample_Validation, 'NegativeSample_Validation_PBI.csv')
    NegativeSample_Test = []
    ReadMyCsv1(NegativeSample_Test, 'NegativeSample_Test_PBI.csv')

    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)
    x_train_pair.extend(NegativeSample_Train)

    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)

    x_test_pair = []
    x_test_pair.extend(PositiveSample_Test)
    x_test_pair.extend(NegativeSample_Test)

    x_train_1_Behavior1, x_train_2_Behavior1 = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior1)
    x_validation_1_Behavior1, x_validation_2_Behavior1 = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior1)
    x_test_1_Behavior1, x_test_2_Behavior1 = GenerateBehaviorFeature(x_test_pair, AllNodeBehavior1)

    x_train_1_Behavior2, x_train_2_Behavior2 = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior2)
    x_validation_1_Behavior2, x_validation_2_Behavior2 = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior2)
    x_test_1_Behavior2, x_test_2_Behavior2 = GenerateBehaviorFeature(x_test_pair, AllNodeBehavior2)

    y_train_Pre = MyLabel(x_train_pair)  # Label->one hot
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train_Pre, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_Pre, num_classes)
    y_test = keras.utils.to_categorical(y_test_Pre, num_classes)

    # ———————————————————— 5 times —————————————————————
    CounterT = 0
    while CounterT < 5:
        # ———————————————————— define ————————————————————

        # ——输入1——
        input1 = Input(shape=(len(x_train_1_Behavior1[0]),), name='input1')
        x1 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.0001))(input1)

        # ——输入2——
        input2 = Input(shape=(len(x_train_2_Behavior1[0]),), name='input2')
        x2 = Dense(256, activation='relu', activity_regularizer=regularizers.l2(0.0001))(input2)

        # ——输入3——
        input3 = Input(shape=(len(x_train_1_Behavior2[0]),), name='input3')
        x3 = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.0001))(input3)

        # ——输入4——
        input4 = Input(shape=(len(x_train_2_Behavior2[0]),), name='input4')
        x4 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.0001))(input4)

        # ——连接——
        flatten = keras.layers.concatenate([x1, x2, x3, x4])

        # ——隐藏层
        hidden = Dense(512, activation='relu', name='hidden1', activity_regularizer=regularizers.l2(0.0001))(flatten)
        hidden = Dropout(rate=0.3)(hidden)
        hidden = Dense(256, activation='relu', name='hidden2', activity_regularizer=regularizers.l2(0.0001))(hidden)
        hidden = Dropout(rate=0.2)(hidden)
        hidden = Dense(128, activation='relu', name='hidden3', activity_regularizer=regularizers.l2(0.0001))(hidden)
        # # hidden = Dropout(rate=0.1)(hidden)
        # hidden = Dense(1024, activation='relu', name='hidden4', activity_regularizer=regularizers.l2(0.0001))(hidden)


        # ——输出层——
        output = Dense(num_classes, activation='softmax', name='output')(hidden)  # category
        model = Model(inputs=[input1, input2, input3, input4], outputs=output)

        # 打印网络结构
        model.summary()

        # ——编译——
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # —————————————————————— train ——————————————————————
        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                      patience=10, mode='auto')  # Automatically adjust the learning rate
        history = model.fit(
            {'input1': x_train_1_Behavior1, 'input2': x_train_2_Behavior1,
             'input3': x_train_2_Behavior2, 'input4': x_train_2_Behavior2,
             }, y_train,

            validation_data=({'input1': x_validation_1_Behavior1, 'input2': x_validation_2_Behavior1,
                              'input3': x_validation_1_Behavior2, 'input4': x_validation_2_Behavior2,
                              },
                             y_validation), callbacks=[reduce_lr], epochs=50, batch_size=64)

        # —————————————————————— 训练模型 ——————————————————————
        ModelName = 'my_model' + str(CounterT) + '.h5'
        model.save(ModelName)  # 保存模型

        # 输出预测值
        ModelTest = Model(inputs=model.input, outputs=model.get_layer('output').output)
        ModelTestOutput = ModelTest.predict(
            [x_test_1_Behavior1, x_test_2_Behavior1,
             x_test_1_Behavior2, x_test_2_Behavior2,
             ])

        print(ModelTestOutput.shape)
        print(type(ModelTestOutput))
        # StorFile(ModelTestOutput, 'ModelTestOutput.csv')
        # 输出值为label、1的概率
        LabelPredictionProb = []
        LabelPrediction = []

        counter = 0
        while counter < len(ModelTestOutput):
            rowProb = []
            rowProb.append(y_test_Pre[counter])
            rowProb.append(ModelTestOutput[counter][1])
            LabelPredictionProb.append(rowProb)

            row = []
            row.append(y_test_Pre[counter])
            if ModelTestOutput[counter][1] > 0.5:
                row.append(1)
            else:
                row.append(0)
            LabelPrediction.append(row)

            counter = counter + 1
        LabelPredictionProbName = 'PBI_RealAndPredictionProbA+B' + str(CounterT) + '.csv'
        StorFile(LabelPredictionProb, LabelPredictionProbName)
        LabelPredictionName = 'PBI_RealAndPredictionA+B' + str(CounterT) + '.csv'
        StorFile(LabelPrediction, LabelPredictionName)

        CounterT = CounterT + 1
