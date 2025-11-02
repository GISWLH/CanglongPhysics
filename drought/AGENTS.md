运行代码所需的环境在conda activate torch，因此运行py代码前先激活环境
timeout xx其中xx应该大于10分钟，因为代码运行较慢，可以多给一些时间
我不喜欢定义太过复杂的函数，并运行main函数，我是深度jupyter notebook用户，我喜欢直接的代码，简单的函数定义是可以接受的
使用matplotlib可视化，绘图使用Arial字体(在linux中手动增加我们的arial字体），绘图中的图片标记都用英文

这个项目是进行CAS-Canglong AI气象模型的多变量预报与检验。
Z:\Data\hindcast_2022_2023\hindcast_2022_2023_lead1.nc to Z:\Data\hindcast_2022_2023\hindcast_2022_2023_lead6.nc
有提前一周到6周的预测，以hindcast_2022_2023_lead6.nc的时间为准，统计。
- 气候态数据：`E:\data\climate_variables_2000_2023_weekly.nc` (土壤水没有， 请手动计算土壤水的周平均为2022-2023的平均吧)

现在有volumetric_soil_water_layer，total_precipitation，surface_latent_heat_flux，2m_temperature，surface_sensible_heat_flux，2m_dewpoint_temperature，10m_u_component_of_wind，10m_v_component_of_wind，sea_surface_temperature预测数据（CAS-Canglong）以及Obs代表EC真实值

1. 根据PET等变量计算6个nc所有文件的SPEI-4（四周）计算，可以用SEPI包，添加进度条，为Z:\Data\hindcast_2022_2023\hindcast_2022_2023_lead1.nc to Z:\Data\hindcast_2022_2023\hindcast_2022_2023_lead6.nc添加spei/spei_obs variable.

2. 降水距平TCC