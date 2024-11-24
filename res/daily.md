# 2024 11 23
## get 到的 obs 数据类型问题
1. 在env中get到的obs是uint8，因为get到的是row的image图像，所以uint8完全可以进行理解。
2. 但是我们需要segment的时候，输入进去的我们需要是float32的数据类型。
## matpltlib 绘制图像
1. 当我们获得了一个obs或者是已经segment的img的时候，我们需要绘制出来，这时候我们可以使用
2. matpltlib中的pyplot
3. 直接可以调用pyplot中的imshow()
4. 同时我们还可以调用figer控制窗口，axios，关闭坐标轴，titile，图像题目等
## 图像的归一化
二：图像归一化处理

归一化（Normalization）：归一化的目标是找到某种映射关系，将原数据映射到[a,b]区间上。一般a,b会取[−1,1],[0,1]这些组合。

一般有两种应用场景：
1、把数变为(0, 1)之间的小数
2、把有量纲的数转化为无量纲的数

图像归一化最常见的就是最大最小值归一化方法，公式如下：

OpenCV中实现图像最大与最小值归一化的函数如下：
```
normalize(
src, // 表示输入图像， numpy类型
dst, // 表示归一化之后图像， numpy类型
alpha=None, // 归一化中低值 min
beta=None, // 归一化中的高值max
norm_type=None, // 归一化方法，选择最大最小值归一化 NORM_MINMAX，
dtype=None, // 归一化之后numpy数据类型，一般选择cv.CV_32F
mask=None //遮罩层，默认设置为None
```
基于OpenCV实现图像最大最小值归一化的代码演示如下：
```
image = cv.imread("D:/javaopencv/dahlia_4.jpg")
cv.imshow("input", image)
result = np.zeros(image.shape, dtype=np.float32)
cv.normalize(image, result, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
print(result)
cv.imshow("norm", np.uint8(result*255.0))
cv.waitKey(0)
cv.destroyAllWindows()
```
解释
原图与归一化之后的运行结果完全一致，说明归一化不会改变图像本身的信息存储，但是通过打印出来的像素值可以发现，取值范围从0～255已经转化为0～1之间了，这个对于后续的神经网络或者卷积神经网络处理有很大的好处，tensorflow官方给出mnist数据集，全部采用了归一化之后的结果作为输入图像数据来演示神经网络与卷积神经网络。

值得注意：
归一化：缩放仅仅跟最大、最小值的差别有关。
标准化：缩放和每个点都有关系，通过方差（variance）体现出来。与归一化对比，标准化中所有数据点都有贡献（通过均值和标准差造成影响）。

为什么要标准化和归一化？
提升模型精度：归一化后，不同维度之间的特征在数值上有一定比较性，可以大大提高分类器的准确性。
加速模型收敛：标准化后，最优解的寻优过程明显会变得平缓，更容易正确的收敛到最优解。
## plt绘制图像的要求
1. 当图像的像素[0-255]的时候我们使用(uint8)类型，当像素归一化为[0-1]我们使用(float32)类型的
## np数组逻辑比较
`The truth value of an array with more than one element is ambiguous.`
np数组对象在逻辑比较时，不能和普通list一样进行比较
# 2024 11 24
## 进行segment的时候出现了list index out of range
这个问题我原本是以为是因为中间有一个空值，也就是说里面有一个shape是[]这个样子的
```
import pickle
name = 'sim_512/offline_episode_512_30.pkl'
with open(name,'rb') as f:
    data = pickle.load(f)
data[7]['camera'][40]
问题所在这个位置
```
`data[7]['camera'][40]`经过画图可以知道，这个位置agent看到的是一个帘子，此时mask是空值
## logits
未经处理的逻辑值，一般在神经网络中代指从神经网络中输出的row数据，一般来说神经网络的输出还需要进行sigmod等函数的激活，这个函数就是将row数据处理成匹配的数据。
## datasets and dataloader
https://blog.csdn.net/qq_41813454/article/details/134903615
