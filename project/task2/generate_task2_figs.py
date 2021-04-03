import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("task2lookback.txt")

x = []
y = []
y2 = []

for index, row in df.iterrows():
	x.append(float(row["lookback"]))
	y.append(float(row[" acc"]))
	y2.append(float(row[" loss"]))

plt.figure()
plt.scatter(x,y,label="Accuracy")
# plt.scatter(x,y2,label="Loss")
plt.xlabel("Lookback")
plt.ylabel("Accuracy")
# plt.legend()
plt.tight_layout()
plt.savefig("lookback.png")


plt.figure()
plt.scatter(x,y2,label="Accuracy")
# plt.scatter(x,y2,label="Loss")
plt.xlabel("Lookback")
plt.ylabel("Loss")
# plt.legend()
plt.tight_layout()
plt.savefig("loss.png")