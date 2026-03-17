这个文件夹是为了实现业务运行流程，有一个示例，动态推演未来一年的情况  
示例是analysis/operation/example.md，他们是IAP的模式，我们是CAS-Canglong的AI模式，我们也有海洋和大气两个模型。海洋是CAS-Canglong的海洋模式，直接输出之后16个月的预测。  
大气是CanglongPhysics的周尺度，输入前两个周，滚动预测下一周。

业务运行一般要从输入数据开始，因为输入数据几乎是近实时的，要从ECMWF欧洲中心用cdsapi key下载，此外对于两种模型，输入数据要先标准化，mean和std是两套不同的参数，要注意。  
