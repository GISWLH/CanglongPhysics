# 物理损失约束

## 水量平衡约束

首先来看陆地水量平衡,在大流域尺度上,分流域统计水量平衡误差:

$\Delta S_{\text{soilwater}} = P_{\text{land}} - E_{\text{land}} - R$

严格来说,在陆地大流域尺度上,上述等式成立,每两周的水量变化,能写成上述等式形式。方程在整个流域上是闭合的。对整个流域内的所有网格点进行求和或求平均。使用一个流域掩码(mask)来选择相应的网格点:

$\sum_{i \in basin} \Delta S_{basin_i} \cdot A_{\text{grid}} = \sum_{i \in basin} (P_{\text{land}} - E_{\text{land}} - R)_{basin} A_{\text{grid}}$

同样也可以写出海洋水平衡方程,应该也有大尺度水量平衡,代表着从海面发生的蒸发完全等于直接降落在海面上的降水加上从陆地流入海洋的径流(应该忽略内流区):

$\frac{d(S_{\text{ocean}})}{dt} = P_{\text{ocean}} - E_{\text{ocean}} + R$

同样的思路,大气的水分也是平衡的,大气中的水汽总量(可降水量)的变化,等于进入大气的水汽减去离开大气的水汽。输入是来自海洋和陆地的蒸散(总),输出到海洋和陆地的降水。

$\frac{d(S_{atmos})}{dt} = ET_{\text{land}} + E_{\text{ocean}} - P_{\text{ocean}} - P_{\text{land}}$

其中,$\Delta S_{\text{atmos}}$ 为 Total column water vapour 的变化。对于 P、E 的海洋陆地分配,可以在模型的每个预测步中,获取预测的降水和蒸散。将它们与 Land-sea mask (LSM) 相乘来区分陆地和海洋。

$P_{\text{land}} = P \times LSM$
 $P_{\text{ocn}} = P \times (1 - LSM)$

蒸散也同样,如果我们将上述三个平衡方程相加,右侧的所有通量项(P, E, R)都会两两抵消,最终得到

$\frac{d(S_{\text{ocean}})}{dt} + \frac{d(S_{\text{land}})}{dt} + \frac{d(S_{atmos})}{dt} = 0$

这个公式表达了全球总水量守恒:水不会凭空产生或消失,它只是在海洋、陆地和大气这三个水库之间重新分配。让 AI 模型不断学习水量分配关系,也有助于极端事件的预测,如在一个厄尔尼诺事件的年份,陆地(特别是热带地区)可能会经历更多的降水,导致陆地储水量增加($\frac{d}{dt}(S_{\text{land}}) > 0$)。这部分多出来的水主要来自海洋蒸发,常常会导致全球海平面出现暂时下降。

## 能量平衡约束

### 陆地能量平衡

参考比较经典的 SEBAL 地表能量平衡算法,来自蒸散老祖 Bastiaanssen, Wim GM, et al. "A remote sensing surface energy balance algorithm for land (SEBAL). 1. Formulation." Journal of hydrology 212 (1998): 198-212.

陆地表面能量平衡有这个公式:

记 $R_n$ 为地表接收到的净辐射,它可以用来自太阳的能量自上而下的计算:

$R_n = SWD + LWD - SWU - LWU$

SWD, LWD, SWU, LWU 分别是下行短波,下行长波,地表反射的短波,地表热辐射发射的的长波。

地表吸收到 $R_n$ 的辐射后,自下而上有如下的消耗方式:

$R_n = LE + H + G$

LE, H, G 分别是通过蒸散发(相变)传热、感热(热对流)传热以及向下传递的土壤热通量(热传导),这里需要注意记 LE 向上(向大气),H 向上,G 向下(土壤)为正,即绝对值的和。这里比较重要的是土壤热通量 G。这个是土壤自己保存的热量:

$G = -K \frac{\partial T}{\partial z}$

K 是土壤的热导率(W/m·K)和土壤含水量、质地相关。$\frac{\partial T}{\partial z}$ 是土壤的温度梯度。

这其实是基于热传导公式(一维傅里叶定律,https://en.wikipedia.org/wiki/Thermal_conduction)写的。计算 G 需要通过四层土壤温度,然而在大模型中一次输入四层土壤温度是很奢侈的,这里我们进行简化,利用能量守恒定律,流入土壤的热通量 G,必然等于土壤层内部能量(即热量)的变化率。简单来说,有多少能量流进去,土壤的温度就会相应地上升多少。

$G = C_{\text{soil}} \cdot D \cdot \frac{\Delta T_{\text{soil}}}{\Delta t}$

其中 $C_{\text{soil}}$ 是整个土壤层的体积热容($\text{J m}^{-3} \text{K}^{-1}$),它表示使单位体积的土壤温度升高 1K 所需的能量。D 是土壤总深度,即 $D = d_1 + d_2 + d_3 + d_4$,$\frac{\Delta T_{\text{soil}}}{\Delta t}$ 是土壤温度的变化率。在模型中,根据后一时刻的预测值减去前一时刻的得到:

$\frac{\Delta T_{\text{soil}}}{\Delta t} = \frac{T_{\text{soil}}(t+1) - T_{\text{soil}}(t)}{\Delta t_s}$

$C_{\text{soil}}$ 是土壤体积热容赖于土壤类型和土壤含水量,这里的土壤体积热容采取参数化方案,通过戴永久(Dai et al., 2019a, 2019b)等人提供的土壤固体部分体积热容(Volumetric heat capacity of soil solids in a unit soil volume),结合模型的土壤水输出,就可以构建总的体积热容。需要注意的是,在 ERA5 和土壤体积热容中,土壤数据是分层的,而我们的模型尽可能节省显存,不输出分层数据,只输出一个根据深度加权的数据。模型的输入和输出是加权的四层土壤温度,土壤水,相应地,土壤固体体积热容也进行加权:

$cs_{\text{soil}} = \frac{cs_1 d_1 + cs_2 d_2 + cs_3 d_3 + cs_4 d_4}{D}$

其中,根据 ERA5 的分层,$d_1$-$d_4$ 层高度分别为:0.07, 0.21, 0.72 和 1.89m。则总深度为 2.89m

要计算总的土壤体积热容,需要考虑三部分,土壤主要由三部分组成:固体颗粒、水和空气。其中,空气的热容非常小,与水和固体相比可以忽略不计。这里还需要用土壤水和水的体积热容来计算土壤水的体积热容,二者相加就是总热容:

$C_{\text{soil}} = cs_{\text{soil}} + \theta \cdot c_w$

其中 $\theta$ 是土壤的体积含水量($\text{m}^3 \text{m}^{-3}$),在模型中通过加权的 Volumetric soil water(SW_bulk)计算,$c_w$ 是水的体积热容,是一个常数($4.184 \times 10^6 \text{ J m}^{-3} \text{K}^{-1}$)。最终通过所有上式,我们可以得到土壤热通量的变化,构建陆地能量平衡:

$G = C_{\text{soil}} \cdot D \cdot \frac{T_{\text{soil}}(t+1) - T_{\text{soil}}(t)}{\Delta t_s}$

### 海洋能量平衡

类似陆地,海洋表面也需要遵守能量守恒,同样记海面接收到的能量为

$R_n = SWD + LWD - SWU - LWU$

SWD-SWU 也就是你公式里的 ssr, 海洋表面的净短波辐射,LWD-LWU 就是净长波 str。

这个吸收到的 $R_n$,同样的一部分用于海水蒸发潜热 LE,即你公式里的 slhf,一部分是海水加热大气(上为正)/被大气加热(上为负),sshf,还有一部分向下传递,也可以说是海洋自己储存了热量,类似于上述的土壤热通量 G,因此这个 $R_n$ 还可以写作

$R_n = slhf + sshf + G$

这个 G 其实就是你公式里的 $Q_{\text{ohc}}$,唯一的区别就是,G 的单位是 $\text{W}/\text{m}^2$,$Q_{\text{ohc}}$ 在时间上积分了,单位是 $\text{J}/\text{m}^2$,但是大伙都乘以时间了,没区别。

因此通过上式移项,有:

$Q_{\text{net}} = R_n - slhf - sshf = ssr + str - slhf - sshf$

你的大公式是对的

细节来说,通过自下而上,$Q_{\text{net}}$,也就是海洋的"土壤热通量 G",可以通过一维傅里叶写成(水的热传导率固定为比热容*密度,单位就转换为了 J/m²/K)

$Q_{\text{net}} = \rho_w c_w \int_{z=0}^{Z} \frac{\Delta T}{\Delta t} dz$

上式其实就是海洋每一层的温度变化乘以比热容和密度的垂直积分,需要深层海温数据。但是海洋随着深度的温度变化我们没有数据。你公式里用的方法是上下时刻的海表温度随时间变化乘以比热容来计算的,其实是海表的储热变化,如下

$Q_{\text{net}}^{\text{sfc}} = \rho C_p h \frac{SST_{t+1} - SST_t}{\Delta t}$

因此,这个公式缺少了"海表向下传热"的部分,并没有守恒。特别是 t 和 t+1 时刻离得比较远的时候。例如,如果 t 时刻温度为 10 度,t+1 时刻也是 10 度,海表储热是没变化的,但是其实可能已经将热量传递给下层了,表面由于下雨别的因素导致温度没变。只有一种情况这个公式是等号,即 t 和 t+1 相距非常短,热量没来得及向下传递。还是需要深层海洋温度的,起码是混合层(和大气有热量交换的深度)的温度数据,这个深度不是很浅。例如下面文章里这个深度设置为了 45m(https://link.springer.com/article/10.1007/s00382-019-04697-1)

虽然有一些方法可以反算(eg. equilibrium temperature model)温度但是太复杂了。不一定有梯度了。本想长时间尺度上是否可以忽略这个变化,但是被一篇 science 还有 NC 拦住了,题目 Warming of the World Ocean 和 Deep ocean warming-induced El Niño changes"尽管表面温度在短期内变化较快,但深层海洋吸收了大量的热量,且这种热量的变化不会迅速释放,因此深层储热对于长期气候变化的影响是不可忽视的",可惜。甚至维基也说了这个事情 https://en.wikipedia.org/wiki/Ocean_heat_content

但是还有一招,你可以说短时间上,向下传播的热太少了,对比起辐射、潜热、感热等。因此我的建议是,下载 EC 海洋再分析(https://cds.climate.copernicus.eu/datasets/reanalysis-oras5?tab=overview),看一下滑动的一周时间(你的预报间隔)上,海洋深层热通量对比 ERA5 的感热、潜热、太阳辐射占比,如果很小,你就说,短时间上忽略

如果这样,即可用 ERA5 的净长波 slhf,净短波 sshf,你算的海温 T 来近似的构建这个等式。审稿人问了也有话说了。

## 空气动力学约束

通过静力平衡方程,考虑一个具有单位水平横截面积的垂直空气柱,构建流体静力学方程,在静力平衡条件下有:

$\frac{\partial p}{\partial z} = -\rho g = -\frac{pg}{R_d T_v}$

通常,大气总处于静力平衡状态,静力学方程可以用来粗略地估算气压与高度间的定量关系,或者用于将地面气压订正为海平面气压。这样就直接通过物理约束来控制从表面层到高空层的温度和气压关系。对流体静力学方程移项并两边积分:

$\Phi_2(p_2) - \Phi_1(p_1) = -\int_{p_1}^{p_2} \frac{R_d T_v}{p} dp$

就得到压高方程,倘若假定气体常数为固定 $R_d$,假定在这个高度差内忽略重力加速度的变化和气温的变化:

$\Delta\Phi = R_d \cdot T_v \cdot \ln\left(\frac{p_1}{p_2}\right)$

其中 $\Delta\Phi$ 是位势厚度(Geopotential Thickness),它与几何厚度 $\Delta Z$ 的关系是:

$\Delta\Phi = g \cdot \Delta Z$

在每个高空层,相邻高度场(200, 300, 500, 700, 850 hpa)都可以进行关系修正,例如 850hPa 与 700hPa 层可以计算静力平衡:

$\Phi_{850} - \Phi_{700} = R_d \cdot \frac{T_{850} + T_{700}}{2} \cdot \ln\left(\frac{850}{700}\right)$

假设我们有 N 个气压层(这里为 5),按压强从大到小(高度从低到高)索引为 i=1,2,...,N。

$\Phi_{p_{i+1}} - \Phi_{p_i} = R_d \cdot \frac{T_i + T_{i+1}}{2} \cdot \ln\left(\frac{p_{i+1}}{p_i}\right)$

同样的,在表面层,通过 DEM 可以建立海平面气压与地面气压的关系,只需在压高方程两边取指数:

$p_{\text{sfc}} = p_{\text{msl}} \times \exp\left(-\frac{gz}{R_d T_v^{2m}}\right)$

这样,通过五层空气动力学层层约束,让神经网络学习到大气层级的基本关系

## 气温局地变化方程

热力学能量方程描述了大气中任意一点的温度是如何随时间变化的,综合了动力过程(风的输送)和热力过程(加热/冷却)。可以让神经网络学习到大气运动和温度变化之间最基本的因果关系,而不仅仅是统计相关性。这对于提高模型(尤其是降水预测模型,因为降水与上升运动和潜热释放密切相关)的物理真实性至关重要。

$\frac{\partial T}{\partial t} = -V_h \cdot \nabla_h T - \left(\frac{R_d T}{c_p p} - \frac{\partial T}{\partial p}\right) \omega + \frac{1}{c_p}\frac{dQ}{dt}$

其中 $\frac{\partial T}{\partial t}$ 为气温局地变化项,可用温度的随时间的变化率表示,实际计算时可用多层 Temperature 来计算:

$\frac{\partial T}{\partial t} = \frac{T(t+1) - T(t)}{\Delta t_s}$

其中 $-V_h \cdot \nabla_h T$ 为水平平流项,代表水平方向的风将不同温度的空气输送过来,导致局地温度变化(例如,冷空气平流导致降温)。其中 $V_h$ 是水平风场向量,$\nabla_h T$ 是水平温度梯度,需要在格点数据上通过数值差分来计算,例如在 x 方向的梯度如下,在实际计算中需要多层的 t, u, v 等分量。

$\frac{\partial T}{\partial x} = \frac{T(i+1,j) - T(i-1,j)}{2\Delta x}$

其中 $\left(\frac{R_d T}{c_p p} - \frac{\partial T}{\partial p}\right) \omega$ 是绝热垂直运动项,空气垂直运动引起的温度变化。上升的空气绝热膨胀冷却,下沉的空气绝热压缩增温。这是天气变化(如云的形成和消散)的核心驱动力。在计算时 $\omega = \frac{dp}{dt}$ 是压强坐标系下的垂直速度。$\frac{\partial T}{\partial p}$ 是温度随压强的垂直梯度,同样通过数值差分计算。直接通过多层的 Vertical velocity 和 Temperature 计算。

## 非绝热加热项

在气温局地变化方程中,最后一项 $\frac{1}{c_p}\frac{dQ}{dt}$ 非绝热加热项是难点,因为它难以写为直接的函数,ERA5 并不提供总非绝热加热率变量。这个变量是数值天气预报模式内部通过复杂的物理过程参数化方案(辐射方案、云微物理方案、边界层方案等)计算得出的诊断量。直接从 ERA5 的标准输出变量中精确地重构 $\frac{dQ}{dt}$ 的垂直廓线是极其困难的,几乎相当于重建一个模式的物理过程部分。

但是,也可以忽略这一项,在自由大气中(边界层以上)、晴空区域,非绝热过程(尤其是潜热和感热)相比于平流和垂直运动项要小一个量级。我们可以构建一个忽略非绝热项的损失函数。考虑到这一项和降水高度相关,因此我们打算用一种简化关系的形式,可以用 ERA5 的其他变量来近似主要的非绝热过程。

总的非绝热加热率(下文简称 Q)可以分解为三个主要分量的和:

$Q = Q_{\text{rad}} + Q_L + Q_{\text{sen}}$

**辐射加热** ($Q_{\text{rad}}$): 指大气中的气体分子(如水汽、臭氧、二氧化碳)、云和气溶胶吸收太阳短波辐射和吸收/放出地球长波辐射所造成的净加热或冷却。

**潜热加热** ($Q_L$): 来源于大气中水的三相变化。水汽凝结、凝华释放大量热量,是大气中一个强大且局地的热源;相反,云滴蒸发、降水融化和升华则会吸收热量,造成冷却。

**感热加热** ($Q_{\text{sen}}$): 指地球表面(陆地或海洋)与大气之间通过分子传导和湍流对流直接交换的热量。这种交换由地表与上方空气的温差驱动,其影响通常局限在行星边界层内。

### 辐射加热项计算

其中总辐射加热率廓线 $Q_{\text{rad}}$ 计算时,先计算大气柱总辐射吸收量,其中大气总短波吸收:

$A_{\text{SW}} = F_{\text{TOA,SW}} - F_{\text{SRF,SW}}$

其中 $F_{\text{TOA,SW}}$ 是 Mean top net short-wave radiation flux,$F_{\text{SRF,SW}}$ 是 Mean surface net short-wave radiation flux;再计算大气总长波吸收:

$A_{\text{LW}} = F_{\text{TOA,LW}} - F_{\text{SRF,LW}}$

其中 $F_{\text{TOA,LW}}$ 是 Mean top net long-wave radiation flux,$F_{\text{SRF,SW}}$ 是 Mean surface net long-wave radiation flux。第二步需要构建复合垂直权重廓线,根据大气中主要辐射活性物质的垂直分布,构建两个无量纲的权重廓线,用于分配 $Q_L$ 和 $Q_{\text{sen}}$ 的总能量。对于短波权重廓线,太阳短波辐射主要被水汽、臭氧和云吸收。该权重廓线由这些物质的浓度线性组合而成:

$W_{\text{SW}}(p) = c_1 \cdot q(p) + c_2 \cdot o_3(p) + c_3 \cdot (q_{\text{cl}}(p) + q_{\text{ci}}(p))$

其中 $q(p)$ 是 Specific humidity,$o_3(p)$ 是 Ozone mass mixing ratio,$q_{\text{cl}}(p)$ 是 Specific cloud liquid water content,$q_{\text{ci}}(p)$ 是 Specific cloud ice water content。$c_1, c_2, c_3$ 是无量纲的经验权重系数,这里直接设置为可学习参数。

对于长波权重廓线,长波辐射的吸收和发射主要由水汽和云主导:

$W_{\text{LW}}(p) = c_4 \cdot q(p) + c_5 \cdot (q_{\text{cl}}(p) + q_{\text{ci}}(p))$

因此,在各个压强高度 $p$,总辐射加热率 $Q_{\text{rad}}$(单位:K/s)由短波(SW)加热项和长波(LW)冷却项组成:

$Q_{\text{rad}}(p) = Q_{\text{rad,SW}}(p) + Q_{\text{rad,LW}}(p)$

其中,短波和长波分量:

$Q_{\text{rad,SW}}(p) = \frac{g_0}{c_p} \cdot A_{\text{SW}} \cdot \frac{W_{\text{SW}}(p)}{\int_{p_{\text{TOA}}}^{p_{\text{sfc}}} W_{\text{LW}}(p') dp'}$

$Q_{\text{rad,LW}}(p) = \frac{g_0}{c_p} \cdot A_{\text{LW}} \cdot \frac{W_{\text{LW}}(p)}{\int_{p_{\text{TOA}}}^{p_{\text{sfc}}} W_{\text{LW}}(p') dp'}$

其中 $g_0$ 是标准重力加速度(约 9.8m/s²),$c_p$ 是干空气定压比热,约 1004 Jkg⁻¹K⁻¹,$\int_{p_{\text{TOA}}}^{p_{\text{sfc}}} W_{\text{LW}}(p') dp'$ 是对权重廓线从大气层顶到地表的垂直积分。在离散的压强层上,这可以近似为求和 $\sum_{i=1}^{N} W(p_i) \Delta p_i$,其中 $\Delta p_i$ 是第 i 层的压强厚度。

### 潜热加热项计算

在计算 $Q_L$ 潜热加热时，整层大气的总潜热加热通量与地表降水率成正比：

$\text{INT } Q_L = L_v \cdot \rho_w \cdot P$

其中 $L_v$ 是水的蒸发潜热（约 $2.5 \times 10^6$ J/kg），$\rho_w$ 是水的密度（$10^3$ kg/m³），$P$ 是以 m/s 为单位的降水率。

接下来需要构建垂直廓线，根据已有研究成果（Nelson et al. 2016）定义两个无量纲的、标准化的垂直廓线形状函数：对流廓线 $S_{\text{conv}}(p)$ 和层云廓线 $S_{\text{strat}}(p)$，对流廓线应该是一个深厚的正值函数，峰值在对流层中层。$S_{\text{strat}}(p)$ 则是一个在融化层以上为正、以下为负的"上重下轻"型函数。这两个函数的垂直积分（对压强 p）都应归一化为 1。

对于两种降水，计算降水权重：在每个网格点和时间步，计算对流降水占总降水的比例，接着计算层云降水比例

$f_{\text{conv}} = \frac{P_{\text{conv}}}{P_{\text{total}}}$

$f_{\text{strat}} = 1 - f_{\text{conv}}$

则混合廓线 $S_{\text{total}}(p)$ 是两种标准廓线的加权平均：

$S_{\text{total}}(p) = f_{\text{conv}} \cdot S_{\text{conv}}(p) + f_{\text{strat}} \cdot S_{\text{strat}}(p)$

最后将第一步计算的积分加热率分配到该垂直廓线上，得到最终的 $Q_L(p)$

$Q_L(p) = \text{INT } Q_L \cdot S_{\text{total}}(p)$

虽然有一些经验廓线可供选择（Nelson et al. 2016），但还是较为粗略，且我们的降水没有那么多层。不妨直接将深度学习函数来代替固定的标准廓线，让模型自己去学习廓线形状，那么这样廓线函数就定义为（Soto et al. 2024）：

$S_{\text{total}}(p) = \text{NN}(p, \text{Atmos State})$

### 感热加热项计算

最后一部分是 $Q_{\text{sen}}$ 感热加热项，感热加热是一个相对更简单的过程，其影响主要局限在近地面的大气边界层内。感热加热是地球表面与紧邻的大气之间通过分子传导和湍流直接交换的热量。当白天地表温度高于空气温度时，地表就会加热大气。这种直接加热效应在近地面最强，并随着高度在行星边界层（PBL）内逐渐减弱。在边界层顶之上，大气基本不受地表感热的直接影响。

首先需要把 Mean surface sensible heat flux 累积量转为通量率：

$F_s = \frac{\text{SSHF}}{\Delta t}$

其中 $F_s$ 代表了地表向大气注入热量的速率，单位是 W/m²，而 SSHF 是感热通量，单位是 J/m²，要除以累计时长 2 周。

第二步需要构建线性递减的加热廓线，这一步相对简单，感热加热率从地表值开始，随几何高度 $z$ 线性递减，在边界层顶（blh 变量）处减为零。则近地表的加热率 $Q_{\text{surface}}$ 为：

$Q_{\text{surface}} = \frac{F_s}{\rho \cdot c_p \cdot \Delta z}$

其中，$\rho$ 是近地表空气密度，$c_p$ 是空气定压比热，$\Delta z$ 是最底层的高度厚度，可由 geopotential 计算得到。对于任何几何高度 $z$：

$Q_{\text{sen}}(z) = \begin{cases} Q_{\text{surface}} \cdot \left(1 - \frac{z}{\text{blh}}\right) & z \leq \text{blh} \\ 0 & z > \text{blh} \end{cases}$

------

# 纳维-斯托克斯方程

通过将理想情况 Navier-Stokes 方程嵌入神经网络，可以构建一个简化的水平动量方程作为 PDE 约束，其中原始的 N-S 方程为：

$\frac{\partial u}{\partial t} + (u \cdot \nabla)u = -\frac{1}{\rho}\nabla p + v\nabla^2 u + F$

其中 $u$ 看作三维风场矢量，代表流体 $(u, v, w)$，$\rho$ 是空气密度，$p$ 是压强，$v$ 是运动粘度，$F$ 是外力（科氏力，重力，摩擦力）。对于纬向风（u）和经向风（v），简化的方程如下：

## u 分量（东-西方向）

$\frac{\partial u}{\partial t} + \left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z}\right) = fv - \frac{1}{\rho}\frac{\partial p}{\partial x} + F_x$

## v 分量（南-北方向）

$\frac{\partial v}{\partial t} + \left(u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + w\frac{\partial v}{\partial z}\right) = -fu - \frac{1}{\rho}\frac{\partial p}{\partial y} + F_y$

$(u, v, w)$ 变量分别为 U-component of wind，V-component of wind，Vertical velocity。时间导数 $\frac{\partial u}{\partial t}$ 和 $\frac{\partial v}{\partial t}$ 代表风速随时间的变化率，可以用前后两个时间步的差分来近似 $(u_{t+1} - u_t)/\Delta t$。

其中平流项 $u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + w\frac{\partial u}{\partial z}$ 描述了风场自身的输运，在预测网格中使用数值差分计算空间梯度。

$fv$ 和 $-fu$ 是科氏力，地球自转引起的最重要的"视相力"之一，其中 $f = 2\Omega\sin(\phi)$，其中 $\Omega$ 是地球自转角速度，$\phi$ 是纬度。

气压梯度力 $-\frac{1}{\rho}\frac{\partial p}{\partial x}$ 是驱动大气运动的根本动力，通过 Surface pressure 和 Geopotential 然后通过数值差分计算空间梯度。

最后一项 $F$ 是摩擦力项，代表了地表拖曳和湍流耗散。可以通过 ERA5 的参数化变量计算 Mean eastward turbulent surface stress，Mean northward turbulent surface stress。

N-S 方程在理想情况下成立，但在实际大气情况复杂，因此添加一个误差项。

## 参考文献

Dai, Y., N. Wei, H. Yuan, S. Zhang, W. Shangguan, S. Liu, and X. Lu (2019a), Evaluation of soil thermal conductivity schemes for use in land surface modelling, J. Adv. Model. Earth System, accepted.

Dai, Y., Q. Xin, N. Wei, Y. Zhang, W. Shangguan, H. Yuan, S. Zhang, S. Liu, and X. Lu (2019b), A global high-resolution dataset of soil hydraulic and thermal properties for land surface modeling, J. Adv. Model. Earth System, accepted.