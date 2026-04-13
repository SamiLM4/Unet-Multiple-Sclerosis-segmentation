import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("training_history.csv")

plt.plot(df["train_loss"], label="train")
plt.plot(df["val_loss"], label="val")
plt.legend()
plt.show()