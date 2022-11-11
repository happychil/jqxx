# 本程序基于TensorFlow训练了一个神经网络模型来对运动鞋和衬衫等衣物的图像进行分类。
# 使用tf.keras （高级API）在TensorFlow中构建和训练模型。

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import csv
# 查看当前tensorflow版本
print("当前tensorflow版本", tf.__version__)

# 【1 导入Fashion MNIST数据集】
'''
加载数据集将返回四个NumPy数组：
train_images和train_labels数组是训练集 ，即模型用来学习的数据。
针对测试集 ， test_images和test_labels数组对模型进行测试 
'''
'''
图像是28x28 NumPy数组，像素值范围是0到255。 标签是整数数组，范围是0到9。这些对应于图像表示的衣服类别 ：
标签	    类
0	    T恤
1	    裤子
2	    套衫/卫衣
3	    连衣裙
4	    外衣/外套
5	    凉鞋
6	    衬衫
7	    运动鞋
8	    袋子
9	    短靴/脚踝靴
'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 每个图像都映射到一个标签
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 【2 探索数据】
# 在训练模型之前，让我们探索数据集的格式。下图显示了训练集中有60,000张图像，每个图像表示为28 x 28像素
print("训练集总图片数：", train_images.shape)

# 训练集中有60,000个标签
print("训练集中标签数:", len(train_labels))

# 每个标签都是0到9之间的整数
print("标签取值：", train_labels)

# 测试集中有10,000张图像。同样，每个图像都表示为28 x 28像素
print("测试集总图片数：", test_images.shape)

# 测试集包含10,000个图像标签
print("测试集标签数：", len(test_labels))

# 【3 预处理数据】
# 在训练网络之前，必须对数据进行预处理。如果检查训练集中的第一张图像，将看到像素值落在0到255的范围内
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 将这些值缩放到0到1的范围，然后再将其输入神经网络模型。为此，将值除以255。以相同的方式预处理训练集和测试集非常重要：
train_images = train_images / 255.0
test_images = test_images / 255.0

# 为了验证数据的格式正确，并且已经准备好构建和训练网络，让我们显示训练集中的前36张图像，并在每张图像下方显示班级名称。
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 【4 建立模型】
# 建立神经网络需要配置模型的各层，然后编译模型
# 搭建神经网络结构 神经网络的基本组成部分是层 。图层（神经网络结构）从输入到其中的数据中提取表示
# 深度学习的大部分内容是将简单的层链接在一起。大多数层（例如tf.keras.layers.Dense ）具有在训练期间学习的参数。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

'''
编译模型
在准备训练模型之前，需要进行一些其他设置。这些是在模型的编译步骤中添加的：
损失函数 -衡量训练期间模型的准确性。您希望最小化此功能，以在正确的方向上“引导”模型。
优化器 -这是基于模型看到的数据及其损失函数来更新模型的方式。
指标 -用于监视培训和测试步骤。以下示例使用precision ，即正确分类的图像比例。
'''
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 【5 训练模型】
'''
训练神经网络模型需要执行以下步骤：
1.将训练数据输入模型。在此示例中，训练数据在train_images和train_labels数组中。
2.该模型学习关联图像和标签。
3.要求模型对测试集进行预测（在本示例中为test_images数组）。
4.验证预测是否与test_labels数组中的标签匹配。
'''
# 要开始训练，请调用model.fit方法，之所以这么称呼是因为它使模型“适合”训练数据：
model.fit(train_images, train_labels, epochs=10)

# 比较模型在测试数据集上的表现
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 作出预测 通过训练模型，您可以使用它来预测某些图像。模型的线性输出logits 。附加一个softmax层，以将logit转换为更容易解释的概率。
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# print(predictions[i])
    # print(np.argmax(predictions[i]))
    # print('------------------------------------------------------------------------')
    # 写入多行用writerows
# #python2可以用file替代open
list=[]
for i in range(10000):

    max=np.argmax(predictions[i])
    list.append([i,max])
# print(list)

with open("jg.csv","w",newline='') as csvfile:
    writer = csv.writer(csvfile)

    #先写入columns_name
    writer.writerow(["图片名","对应标签"])
    #写入多行用writerows
    writer.writerows(list)

