# 音频处理后端服务

[🌐 在线演示网站](https://www.mylus.cn)

这是一个基于FastAPI的音频处理后端服务，提供了一系列专业的音频分析功能，包括音频参数计算、波形分析、相关性分析等功能。每个功能都有对应的前端页面，可以通过导航页面访问。

## 🚀 功能特性

- 支持单声道和立体声音频文件处理
- 提供多种音频参数计算
- 支持音频文件相关性分析
- 支持批量文件处理
- 提供CSV格式的结果导出
- 提供友好的Web界面

## 🌐 导航页面

访问根路径 `/` 或 `/static/index.html` 可以进入导航页面，该页面提供了所有功能的入口：

- 音调分析
- C80分析
- 相关性切割
- 相关性合并
- 互相关分析
- RMS分析
- 响度分析
- 粗糙度分析
- 锐度分析

## 📊 API接口说明

### 音频参数计算

#### 1. RMS计算
- **接口**: `/audio_param_calculate/RMS`
- **前端页面**: `/static/rms.html`
- **方法**: POST
- **功能**: 计算音频文件的RMS值（包括总RMS和A计权RMS）
- **输入**: 音频文件（.wav）
- **输出**: CSV文件，包含左右声道的RMS值

#### 2. C80计算
- **接口**: `/audio_param_calculate/C80`
- **前端页面**: `/static/c80.html`
- **方法**: POST
- **功能**: 计算音频文件的C80值（清晰度指标）
- **输入**: 音频文件（.wav）
- **输出**: CSV文件，包含左右声道的C80值

#### 3. 音调性计算
- **接口**: `/audio_param_calculate/tonality`
- **前端页面**: `/static/tonality.html`
- **方法**: POST
- **参数**:
  - `st_calib`: 校准值（默认94）
  - `dBSPL`: SPL值（默认-25）
- **功能**: 计算音频文件的音调性
- **输出**: CSV文件，包含左右声道的音调性值

#### 4. 响度计算
- **接口**: `/audio_param_calculate/loudness`
- **前端页面**: `/static/loudness.html`
- **方法**: POST
- **参数**:
  - `st_calib`: 校准值（默认94）
  - `dBSPL`: SPL值（默认-25）
  - `field_type`: 声场类型（"free"或"diff"，默认"free"）
- **功能**: 计算音频文件的响度值
- **输出**: CSV文件，包含左右声道的响度值

#### 5. 粗糙度计算
- **接口**: `/audio_param_calculate/roughness`
- **前端页面**: `/static/roughness.html`
- **方法**: POST
- **参数**:
  - `st_calib`: 校准值（默认94）
  - `dBSPL`: SPL值（默认-25）
- **功能**: 计算音频文件的粗糙度值
- **输出**: CSV文件，包含左右声道的粗糙度值

#### 6. 锐度计算
- **接口**: `/audio_param_calculate/sharpness`
- **前端页面**: `/static/sharpness.html`
- **方法**: POST
- **参数**:
  - `st_calib`: 校准值（默认94）
  - `dBSPL`: SPL值（默认-25）
  - `weighting`: 权重类型（默认"din"）
  - `field_type`: 声场类型（默认"free"）
- **功能**: 计算音频文件的锐度值
- **输出**: CSV文件，包含左右声道的锐度值

### 音频文件处理

所有音频文件处理接口的前缀为 `/audio_files_deal`

#### 1. 互相关分析
- **接口**: `/audio_files_deal/cross_correlation`
- **前端页面**: `/static/cross_correlation.html`
- **方法**: POST
- **功能**: 计算两个音频文件的互相关，并生成可视化图表
- **输入**: 两个.wav文件
- **输出**: 互相关分析图（PNG格式）

#### 2. 相关性切割
- **接口**: `/audio_files_deal/correlation_cut`
- **前端页面**: `/static/correlation_cut.html`
- **方法**: POST
- **参数**:
  - `max_correlation`: 最大相关系数阈值（默认100）
- **功能**: 根据参考文件对音频进行相关性切割
- **输入**: 参考文件（.wav）和待处理文件（.wav或.zip）
- **输出**: 切割后的音频文件（ZIP格式）

#### 3. 相关性合并
- **接口**: `/audio_files_deal/correlation_merge`
- **前端页面**: `/static/correlation_merge.html`
- **方法**: POST
- **功能**: 基于相关性的音频合并工具
- **输入**: 待合并的音频文件
- **输出**: 合并后的音频文件

## 🛠️ 安装说明

1. 克隆项目
```bash
git clone [项目地址]
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行服务
```bash
python main.py
```

服务启动后，可以通过浏览器访问 `http://localhost:8000` 进入导航页面。

## 📝 注意事项

- 所有音频文件处理接口支持.wav格式
- 部分接口支持.zip格式的批量文件处理
- 对于立体声音频，默认会分别处理左右声道
- 所有数值结果保留两位小数
- 部分计算需要较高的采样率（如响度计算需要48kHz以上）

## 🤝 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 📄 许可证

[许可证类型]
