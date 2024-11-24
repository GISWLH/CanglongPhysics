# CanglongPhysics

This project tried to add physics information in AI model.  

We want to build model like below:

![](https://imagecollection.oss-cn-beijing.aliyuncs.com/office/20241124203548.png)

## Physical information

### Data scale  

一种技巧是学习原始数据的Scale

1. 数据准备

- 假设您有 40 年的 ERA5 降水数据，空间分辨率为 721×1440（对应于全球覆盖的网格点）。
- 将这 40 年的数据按小时、日或月等时间步长读取，具体时间步取决于预测需求（例如，逐小时或逐日）。

2. 计算降水差值（趋势）

- 对于每个网格点 $ (i,j)(i, j)(i,j) $$和每个时间点 ttt，计算降水量的时间差值（即变化率）。假设时间步长为 Δt\Delta tΔt（例如1小时或1天），差值计算公式为： 

$$

\Delta P_{i,j,t} = P_{i,j,t+\Delta t} - P_{i,j,t}
$$



- 这个过程会为每个网格点和每个时间点生成一组降水差值数据。

3. 汇总所有网格点和时间的差值数据

- 将 40 年的所有差值数据汇总，这样得到的差值数据集合将包括所有网格点和时间点上的降水差值量（趋势）。
- 假设您有 N个时间点（例如 40 年逐日数据约为 14600 天），那么每个网格点上将会有 N个差值。

4. 计算差值数据的标准差

- 将所有网格点和时间点的降水差值放在一起，计算其标准差，以此获得降水趋势的“典型变化尺度”。 
  $$
  \sigma_{\Delta P} = \sqrt{\frac{1}{N} \sum_{k=1}^N (\Delta P_k - \bar{\Delta P})^2}
  $$
  

5. 生成缩放因子

- 根据 NeuralGCM 的方法，将标准差缩小为 0.01 倍，作为降水趋势的缩放因子。 降水缩放因子:

$$
\text{降水缩放因子} = 0.01 \times \sigma_{\Delta P}
$$



6. 应用缩放因子

- 在训练过程中，将深度学习模块输出的降水趋势与该缩放因子相乘，使输出值符合真实降水变化的物理尺度。

### PINN physics

some useful link:

[Navier-Stokes by PINNs || Physics Informed Reinforcement Learning || Seminar on_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Ej411Y7A2/?vd_source=5cd6007d3cde7a1d36157e015bd4aef0)

[Scien42/NSFnet：用于 2D 不可压缩 Navier-Stokes 方程的 PINN](https://github.com/Scien42/NSFnet)

[纳维-斯托克斯（Navier-Stokes）方程的推导 – 四都教育](https://sudoedu.com/数学物理方程视频课程/数学物理方程的导出/纳维-斯托克斯（navier-stokes）方程的推导/)

### Codebook

Another useful model could be codebook

Esser, P., Rombach, R., & Ommer, B. (2021). *Taming Transformers for High-Resolution Image Synthesis* (No. arXiv:2012.09841). arXiv. https://doi.org/10.48550/arXiv.2012.09841

1. **特征表示的离散化**：
   - 原始的连续特征图（如降水数据的特征图）中的每个位置都可能有大量的细节变化和微小噪声。通过向量量化，特征图中每个位置的连续特征向量被离散化到固定的代码本集合中，从而减少特征中的随机波动，形成更具有代表性的**离散特征**。
   - 离散化后，模型更关注主要特征模式，而忽略噪声，这在降水数据中可能有助于识别重要的降水模式或结构。
2. **特征压缩**：
   - 向量量化将每个位置的 64 维特征向量限制在代码本集合中，代码本集合的大小通常远小于特征向量的所有可能取值。因此，可以在保持主要特征的前提下，显著减少特征数据的总信息量。
   - 在全球降水数据处理中，数据量往往较大，特征压缩能减小存储和计算成本。例如，在后续的解码或进一步分析中，使用离散特征可以减小模型的复杂度，提升处理效率。
3. **有助于生成模型的控制**：
   - 对于类似全球降水这样的大尺度地理数据，离散化的潜在表示可以使模型更容易控制生成过程。例如在重建全球降水图时，模型只需在代码本的有限向量集合中选择代表性特征向量，而不是在连续空间中搜索所有可能的数值。
   - 这种控制性尤其在生成模型（如 VQGAN）中显得重要，使生成的降水图更符合自然模式，而不是随机或无规律的分布。
4. **更好的泛化能力**：
   - 通过代码本量化得到的离散化特征能够帮助模型对不同区域的降水特征有一致性的表示，从而可能提高对不同地理区域的泛化能力。例如，模型可以学习到某些降水模式对应的特征向量，并在未来生成过程中重用这些模式。
   - 这种泛化能力在极端气候预测和多地同时生成降水分布时具有重要意义。
