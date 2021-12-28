from matplotlib import pyplot as plt
from datetime import datetime

import pyarrow.csv as csv
import seaborn as sns
import pyarrow as pa
import pandas as pd
import numpy as np
import warnings
import string
import random
import os

warnings.simplefilter("ignore")

os.listdir()

path = "data/"

os.listdir(path)


def gen_random_string(length: int = 32) -> str:
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


dt = pd.date_range(
    start=datetime(2000, 1, 1),
    end=datetime(2021, 1, 1),
    freq='min'
)

np.random.seed = 42
df_size = len(dt)
print("\n\n    Quantidade de registros: {}".format(df_size))

df = pd.DataFrame({
    'date': dt,
    'a': np.random.rand(df_size),
    'b': np.random.rand(df_size),
    'c': np.random.rand(df_size),
    'd': np.random.rand(df_size),
    'e': np.random.rand(df_size),
    'str1': [gen_random_string() for x in range(df_size)],
    'str2': [gen_random_string() for y in range(df_size)]
})

df.head()

df.tail()

pd_write_list = []

for pdwl in range(50):
    print("\n\n    Pandas is writing...")
    time_start = datetime.now()
    df.to_csv(path + 'csv_pandas.csv', index=False, sep=';', header=True)
    time_end = datetime.now()
    w_pd_csv = time_end - time_start
    pd_write_list.append(w_pd_csv.seconds)
    print("    Tempo decorrido de escrita .csv: {} ou {} segundos.".format(w_pd_csv, w_pd_csv.seconds))


pd_gz_write_list = []

for pdgwl in range(50):
    time_start = datetime.now()
    df.to_csv(path + 'csv_pandas.csv.gz', index=False, sep=';', header=True, compression='gzip')
    time_end = datetime.now()
    w_pd_csv_gz = time_end - time_start
    pd_gz_write_list.append(w_pd_csv_gz.seconds)
    print("    Tempo decorrido de escrita .csv.gz: {} ou {} segundos.".format(w_pd_csv_gz, w_pd_csv_gz.seconds))


pd_read_list = []

for pdrl in range(50):
    print("\n\n    Pandas is reading...")
    time_start = datetime.now()
    df1 = pd.read_csv(path + 'csv_pandas.csv', sep=';')
    time_end = datetime.now()
    r_pd_csv = time_end - time_start
    pd_read_list.append(r_pd_csv.seconds)
    print("    Tempo decorrido de leitura .csv: {} ou {} segundos.".format(r_pd_csv, r_pd_csv.seconds))


pd_gz_read_list = []

for pdgrl in range(50):
    time_start = datetime.now()
    df2 = pd.read_csv(path + 'csv_pandas.csv.gz', sep=';')
    time_end = datetime.now()
    r_pd_csv_gz = time_end - time_start
    pd_gz_read_list.append(r_pd_csv_gz.seconds)
    print("    Tempo decorrido de leitura .csv.gz: {} ou {} segundos.".format(r_pd_csv_gz, r_pd_csv_gz.seconds))


df_pa = df.copy()
df_pa['date'] = df_pa['date'].values.astype(np.int64) // 10 ** 9

df_pa.head()

df_pa.tail()

df_pa_table = pa.Table.from_pandas(df_pa)


pa_write_list = []

for pawl in range(50):
    print("\n\n    PyArrow is writing...")
    time_start = datetime.now()
    csv.write_csv(df_pa_table, path + 'csv_pyarrow.csv')
    time_end = datetime.now()
    w_pa_csv = time_end - time_start
    pa_write_list.append(w_pa_csv.seconds)
    print("    Tempo decorrido de escrita .csv: {} ou {} segundos.".format(w_pa_csv, w_pa_csv.seconds))


pa_gz_write_list = []

for pagwl in range(50):
    with pa.CompressedOutputStream(path + 'csv_pyarrow.csv.gz', 'gzip') as out:
        time_start = datetime.now()
        csv.write_csv(df_pa_table, out)
        time_end = datetime.now()
        w_pa_csv_gz = time_end - time_start
        pa_gz_write_list.append(w_pa_csv_gz.seconds)
        print("    Tempo decorrido de escrita .csv.gz: {} ou {} segundos.".format(w_pa_csv_gz, w_pa_csv_gz.seconds))


pa_read_list = []

for parl in range(50):
    print("\n\n    PyArrow is reading...")
    time_start = datetime.now()
    df_pa_1 = csv.read_csv(path + 'csv_pyarrow.csv')
    time_end = datetime.now()
    r_pa_csv = time_end - time_start
    pa_read_list.append(r_pa_csv.seconds)
    print("    Tempo decorrido de leitura .csv: {} ou {} segundos.".format(r_pa_csv, r_pa_csv.seconds))


pa_gz_read_list = []

for pagrl in range(50):
    time_start = datetime.now()
    df_pa_2 = csv.read_csv(path + 'csv_pyarrow.csv.gz')
    time_end = datetime.now()
    r_pa_csv_gz = time_end - time_start
    pa_gz_read_list.append(r_pa_csv_gz.seconds)
    print("    Tempo decorrido de leitura .csv.gz: {} ou {} segundos.".format(r_pa_csv_gz, r_pa_csv_gz.seconds))


write = {"write": ["pd_csv", "pd_csv_gz", "pa_csv", "pa_csv_gz"],
         "values": [w_pd_csv.seconds, w_pd_csv_gz.seconds, w_pa_csv.seconds, w_pa_csv_gz.seconds]}
df_write = pd.DataFrame(write)

df_write

sns.barplot(x=df_write["write"], y=df_write["values"], data=df_write)
plt.title("Análise do tempo de escrita entre Pandas e PyArrow")
plt.xlabel("Arquivos Escritos")
plt.ylabel("Tempo da escrita em Segundos")
plt.savefig("images/bar_write.png", dpi=150)
plt.close()

read = {"read": ["pd_csv", "pd_csv_gz", "pa_csv", "pa_csv_gz"],
        "values": [r_pd_csv.seconds, r_pd_csv_gz.seconds, r_pa_csv.seconds, r_pa_csv_gz.seconds]}
df_read = pd.DataFrame(read)

df_read

sns.barplot(x=df_read["read"], y=df_read["values"], data=df_read)
plt.title("Análise do tempo de leitura entre Pandas e PyArrow")
plt.xlabel("Arquivos Lidos")
plt.ylabel("Tempo da leitura em Segundos")
plt.savefig("images/bar_read.png", dpi=150)
plt.close()


plt.plot(pd_write_list)
plt.plot(pa_write_list)
plt.title("Análise do tempo de escrita entre Pandas e PyArrow")
plt.xlabel("Quantidade de Repetições de Escrita")
plt.ylabel("Tempo da escrita em segundos")
plt.savefig("images/plot_write_csv.png", dpi=150)
plt.close()

plt.plot(pd_read_list)
plt.plot(pa_read_list)
plt.title("Análise do tempo de leitura entre Pandas e PyArrow")
plt.xlabel("Quantidade de Repetições de Leitura")
plt.ylabel("Tempo da leitura em segundos")
plt.savefig("images/plot_read_csv.png", dpi=150)
plt.close()

plt.plot(pd_gz_write_list)
plt.plot(pa_gz_write_list)
plt.title("Análise do tempo de escrita com compressão entre Pandas e PyArrow")
plt.xlabel("Quantidade de Repetições de Escrita com Compressão")
plt.ylabel("Tempo da escrita com compressão em segundos")
plt.savefig("images/plot_write_gz.png", dpi=150)
plt.close()

plt.plot(pd_gz_read_list)
plt.plot(pa_gz_read_list)
plt.title("Análise do tempo de leitura com compressão entre Pandas e PyArrow")
plt.xlabel("Quantidade de Repetições de Leitura com Compressão")
plt.ylabel("Tempo da leitura com compressão em segundos")
plt.savefig("images/plot_read_gz.png", dpi=150)
plt.close()

# References

# https://towardsdatascience.com/stop-using-pandas-to-read-write-data-this-alternative-is-7-times-faster-893301633475
