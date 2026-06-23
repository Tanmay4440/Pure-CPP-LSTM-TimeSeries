import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prediction.csv")

plt.figure(figsize=(10,5))

plt.plot(
    df["x"],
    df["true"],
    label="True",
    linewidth=3
)

plt.plot(
    df["x"],
    df["predicted"],
    '--',
    label="Predicted",
    linewidth=3
)

plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("Actual vs Predicted")
plt.legend()
plt.grid()

plt.show()