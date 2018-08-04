### -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt

# 円の描画
r = 0.5
t = np.linspace(0,2*np.pi,100)
x = r * np.cos(t)
y = r * np.sin(t)

# 乱数によりサンプル発生
N = 3000 # サンプル数
sample = np.random.rand(N,2) - 0.5
is_inside = np.sqrt(np.sum(sample**2, axis=1)) <= 0.5
color = ['r' if i else 'k' for i in is_inside]

## サンプルプロット
#plt.plot(x,y)
#plt.show()


# サンプルプロット
plt.plot(x,y)
plt.scatter(sample[:,0], sample[:,1], c=color, s=5.0)
plt.title("calculated area: %.4f" % (sum(is_inside)*1. / len(is_inside)))
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.show()

## Nを増やすことによる収束をグラフ化
#div_num = 40
#res = np.zeros(div_num)
#for index, val in enumerate(np.linspace(1,N,div_num)):
#    val = int(val)
#    s = is_inside[:val]
#    res[index] = sum(s)*1./len(s)
#plt.plot(res, marker='.')
#plt.show()

# 結果
print('num samples INSIDE the circle: %d' % sum(is_inside))
print('num samples OUTSIDE the circle: %d' % sum(is_inside==False))
print("calculated area: %.4f" % (sum(is_inside)*1. / len(is_inside)))
print("actual area: %.4f" % (np.pi * r**2))
