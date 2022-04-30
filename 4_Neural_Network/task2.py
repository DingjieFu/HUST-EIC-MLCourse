import os
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn. feature_extraction.text import TfidfVectorizer
torch.set_default_tensor_type(torch.DoubleTensor)


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(10000, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 20)

    def forward(self, x):
        x = F.relu(self.fc1(x))      # output(256)
        x = F.relu(self.fc2(x))      # output(64)
        x = self.fc3(x)              # output(20)
        return x


# .dat文件读取
def dataLoader(trainPath, testPath):
    vectorizer = TfidfVectorizer(max_features=10000)
    vectors_train = None
    vectors_test = None
    with open(trainPath, 'rb') as f:
        train_texts = pickle.load(f)
        vectors_train = vectorizer.fit_transform(train_texts)
    with open(testPath, 'rb') as f:
        test_texts = pickle.load(f)
        vectors_test = vectorizer.fit_transform(test_texts)
    # print(vectors_train.shape, vectors_test.shape)
    return vectors_train.toarray(), vectors_test.toarray()


# 对标签进行onehot编码
def labelOneHot(thisLabel):
    returnVec = [0]*20  # 每一个变成20维, 0-19
    returnVec[thisLabel] = 1
    return returnVec


# 独热标签
def trainLabel(path):
    train_labals = []
    with open(path) as f:
        for line in f.readlines():
            train_labals.append([int(line.strip())])
    return np.array(train_labals)


# 训练网络
def trainNetwork(trainX, trainY, testSize):
    # 这里的test_size=0.25代表选择1/4的数据作为测试集
    x_train, x_test, y_train, y_test = train_test_split(
        trainX, trainY, test_size=testSize)
    print("训练集数据：", x_train.shape, y_train.shape)
    print("测试集数据：", x_test.shape, y_test.shape)
    # print(test_data.shap, test_label.shape)
    # 检查是否有预训练权重
    pretrained_path = "MyNet_best.pth"
    net = MyNet()
    if os.path.exists(pretrained_path):  # 如果预训练模型存在则加载预训练模型参数
        net.load_state_dict(torch.load(pretrained_path))
        print("Load pretrained weights successfully!")
    loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 优化器
    # -------开始训练了-------
    print("----------开始训练了----------")
    for epoch in range(5):  # 训练的轮次
        running_loss = 0.0  # 当前损失
        running_accuracy = 0.0  # 当前准确率
        for step in range(len(x_train)):
            # 一定要转为tensor格式 这是pytorch所要求的
            inputs, labels = torch.tensor(
                x_train[step]), torch.tensor(y_train[step])
            # print("step:", step)
            # print("inputs:", inputs, inputs.shape)
            # print("labels:", labels, labels.shape)
            optimizer.zero_grad()  # 梯度清零
            outputs = net(inputs)  # forward + backward + optimize
            # print("outputs:", outputs, outputs.shape)
            # print(np.argmax(outputs.tolist()))
            outputs = torch.unsqueeze(outputs, 0)
            labels = labels.type(torch.LongTensor)
            # print(outputs)
            # print(labels)
            loss = loss_function(outputs, labels)  # 计算损失
            # print("loss:", loss)
            loss.backward()  # 反向传播 backpropagation
            optimizer.step()  # 进行单次优化 更新所有参数

            # print statistics
            running_loss += loss.item()
            if step % 1000 == 999:  # 每1000个mini-batch预测一次
                rightCnt = 0  # 预测正确的个数
                for i in range(len(x_test)):
                    with torch.no_grad():
                        outputs = net(torch.tensor(x_test[i]))
                        predict_y = np.argmax(outputs.tolist())  # 概率最大标签
                        # print(predict_y, y_test[i])
                        if predict_y == y_test[i]:
                            rightCnt += 1
                accuracy = float(rightCnt)/(i+1)

                save_path = "MyNet_best.pth"
                if accuracy >= running_accuracy:  # 保存精度最高的模型
                    torch.save(net.state_dict(), save_path)
                    running_accuracy = accuracy
                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                running_loss = 0.0

    print('Finished Training!')

    # 训练完才保存参数 因此默认只保存最后一次的训练参数
    save_path = 'MyNet_last.pth'
    torch.save(net.state_dict(), save_path)  # 只保存模型的参数
    # torch.save(net, save_path)  # 保存整个模型


# 对测试集数据预测
def predictTest(testData):
    with open("test_labels.txt", mode="w") as f:
        net = MyNet()
        net.load_state_dict(torch.load('MyNet_best.pth'))
        for i in range(len(testData)):
            with torch.no_grad():
                outputs = net(torch.tensor(testData[i]))
                predict_y = np.argmax(outputs.tolist())  # 概率最大标签
                f.write(str(int(predict_y))+"\n")
                print(predict_y)


if __name__ == '__main__':
    traindatapath = "PythonDeveloper/Machine_Learning/4_Neural_Network/task2/train/train_texts.dat"
    trainlabelpath = "PythonDeveloper/Machine_Learning/4_Neural_Network/task2/train/train_labels.txt"
    testdatapath = "PythonDeveloper/Machine_Learning/4_Neural_Network/task2/test/test_texts.dat"
    train_X, test_X = dataLoader(traindatapath, testdatapath)
    train_Y = trainLabel(trainlabelpath)
    print(train_X.shape, test_X.shape, train_Y[0].shape)
    trainNetwork(train_X, train_Y, 0.05)
    # predictTest(test_X)
