import urllib.request
import pandas as pd
import io

url = "https://psl.noaa.gov/data/correlation/nina34.data"

with urllib.request.urlopen(url) as resp:
    raw = resp.read().decode("utf-8")

lines = raw.strip().splitlines()

# 第一行是年份范围，例如 "1950 2024"
header_line = lines[0].strip().split()
year_start = int(header_line[0])
year_end = int(header_line[1])

rows = []
for line in lines[1:]:
    parts = line.split()
    if len(parts) < 13:
        continue
    year = int(parts[0])
    months = []
    for v in parts[1:13]:
        val = float(v)
        # NOAA缺测值为 -9.99 或 -99.99
        if val <= -9.0:
            months.append("")
        else:
            months.append(round(val, 2))
    rows.append([year] + months)

df = pd.DataFrame(rows, columns=["Year", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

out_path = "analysis/operation/SSTmodel/ENSO_all.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} years ({df['Year'].iloc[0]}–{df['Year'].iloc[-1]}) to {out_path}")
print(df.tail())
