# coding = utf-8
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import paddle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from model.neural import Classifier

df = pd.read_csv('dataset/train.csv')

# corr = df.corr(method='spearman')
# ax = plt.subplots(figsize=(20, 20))
# ax  = sns.heatmap(corr,cmap='Spectral',annot=False)
# plt.savefig('mobile_heat')

df['Activity']=df['Activity'].map({
    'LAYING': 0,
    'STANDING': 1,
    'SITTING': 2,
    'WALKING': 3,
    'WALKING_UPSTAIRS': 4,
    'WALKING_DOWNSTAIRS': 5
})

df_train = df.iloc[:7000]
df_test = df.iloc[7000:]
pd.DataFrame(df_test).to_csv('test.csv')

X_more = df_train[:-1000].sample(n=None, frac=1, replace=False, weights=None, random_state=None, axis=None)
noise = np.random.normal(loc=0, scale=0.003, size=X_more.shape)
noise[:,-1]=0
X_more = X_more + noise
df_train=pd.concat([X_more,df_train],ignore_index=True)
df_train=df_train.sample(frac=1)

print(df_train.shape)
model=Classifier()
opt = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())
# opt = paddle.optimizer.Momentum(learning_rate=0.005, parameters=model.parameters(), weight_decay=0.0002)
loss_fn = paddle.nn.CrossEntropyLoss()
EPOCH_NUM = 1000  # 设置外层循环次数
BATCH_SIZE = 32  # 设置batch大小
training_data = df_train.iloc[:12000].values.astype(np.float32)
val_data = df_train.iloc[12000:].values.astype(np.float32)

training_data = training_data.reshape(-1, 1, 562)
val_data = val_data.reshape(-1, 1, 562)



# 定义外层循环
maxval = 0
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        model.train()
        x = np.array(mini_batch[:, :, :-1])  # 获得当前批次训练数据
        y = np.array(mini_batch[:, :, -1:])  # 获得当前批次训练标签

        # 将numpy数据转为飞桨动态图tensor的格式
        features = paddle.to_tensor(x)

        y = paddle.to_tensor(y)
        # 前向计算
        predicts = model(features)

        # 计算损失
        loss = loss_fn(predicts, y.flatten().astype(int))
        avg_loss = paddle.mean(loss)

        # 反向传播，计算每层参数的梯度值
        avg_loss.backward()
        # 更新参数，根据设置好的学习率迭代一步
        opt.step()
        # 清空梯度变量，以备下一轮计算
        opt.clear_grad()

        # 训练与验证
        if iter_id % 374 == 0 and iter_id!=0:
            acc = predicts.argmax(1) == y.flatten().astype(int)
            acc = acc.astype(float).mean()

            model.eval()
            val_predict = model(paddle.to_tensor(val_data[:, :, :-1])).argmax(1)
            val_label = val_data[:, :, -1]
            val_acc = np.mean(val_predict.numpy() == val_label.flatten())

            print("epoch: {}, iter: {}, loss is: {}, acc is {} / {}".format(
                epoch_id, iter_id, avg_loss.numpy(), acc.numpy(), val_acc))
            if val_acc >= maxval:
                paddle.save(model.state_dict(), 'my_best_model.pdparams')

                maxval=val_acc
                print('best is ',epoch_id)
                # maxval = val_acc
                # model.eval()
                # test_data = paddle.to_tensor(df_test.values.reshape(-1, 1, 561).astype(np.float32))
                # test_predict = model(test_data)
                # test_predict = test_predict.argmax(1).numpy()
                # test_predict = pd.DataFrame({'Activity': test_predict})
                # test_predict['Activity'] = test_predict['Activity'].map({
                #     0: 'LAYING',
                #     1: 'STANDING',
                #     2: 'SITTING',
                #     3: 'WALKING',
                #     4: 'WALKING_UPSTAIRS',
                #     5: 'WALKING_DOWNSTAIRS'})
                # name = 'submission' + str(epoch_id) + ".csv"
                # if val_acc >= 0.99:
                #     test_predict.to_csv(name, index=None)








