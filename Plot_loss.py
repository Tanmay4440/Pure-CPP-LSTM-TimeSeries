import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loss.csv")

plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss vs Epoch")
plt.grid()
plt.show()