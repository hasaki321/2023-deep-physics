更改：
	Linear_size1,Linear_size2 -> L1, L2
	alpha,beta -> alpha,1-alpha
	训练策略默认写两种，程序根据输入自行判断
	测试方式默认两种，程序根据输入自行判断

fix：
	兼容训练设备
	归一化问题
	可以自选 scaler

未完成：
	hid feature
	训练方式 3

注意：
	json 文件中的参数表要统一，没有可以填空字符串
