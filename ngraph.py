import numpy as np
import matplotlib.pyplot as plt


train_add=np.array([18.38721, 10.36628, 5.940838, 5.940838, 5.940838, 5.940838, 5.940838, 5.940838, 5.940838])
test_add=np.array([18.35723, 11.07669, 6.80498, 6.80498, 6.80498, 6.80498, 6.80498, 6.80498, 6.80498])
train_cd=np.array([18.38721, 10.38166, 5.804307, 5.804307, 5.804307, 5.804307, 5.804307, 5.804307, 5.804307])
test_cd=np.array([18.35773, 11.09226, 6.748478, 6.748478, 6.748478, 6.748478, 6.748478, 6.748478, 6.748478])

n = np.linspace(1, 9, 9)

fig, ax = plt.subplots()
ax.plot(n, train_add, color='blue')
ax.plot(n, train_cd, color='red')
ax.plot(n, test_add, color='blue', linestyle='dotted')
ax.plot(n, test_cd, color='red', linestyle='dotted')


ax.legend(['train_add', 'train_cd', 'test_add','test_cd'])
ax.set_title('n order perplexity')
plt.savefig('ngrams.png')