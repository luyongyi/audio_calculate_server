from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from utils.safe_extract import FileExtractor
import utils.waveform_analysis.wave_analyzer  as wave_analyzer
import soundfile as sf
import csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import zipfile
import uuid
import tempfile
import shutil
from utils.waveform_analysis.audio_comparison import (
    calculate_rms_difference, 
    plot_rms_difference,
    AudioCompareParams,
    highpass_filter
)
from utils.waveform_analysis._common import load
plt.switch_backend('Agg')
router = APIRouter()

#计算滑窗点积，也就是两个文件的互相关,并且返回plot图
@router.post("/cross_correlation")
async def cross_correlation(files: UploadFile = File(...),
                            filesRef: UploadFile = File(...)):
    #仅支持两个.wav文件，也是按照extractor的规则
    if files.filename.endswith('.wav') and filesRef.filename.endswith('.wav'):
        file_path=f"files/correlation_{str(uuid.uuid4())}"
        #计算滑窗点积，也就是两个文件的互相关，如果是双声道文件，默认只取左边声道
        os.makedirs(file_path,exist_ok=True)
        data, samplerate = sf.read(files.file)
        data_ref, samplerate_ref = sf.read(filesRef.file)
        #如果是双声道文件，默认只取左边声道
        if data.ndim == 2:
            data = data[:, 0]
        if data_ref.ndim == 2:
            data_ref = data_ref[:, 0]
        #计算滑窗点积，也就是两个文件的互相关
        corr = signal.correlate(data, data_ref, mode='valid')
        #返回plot图,横坐标根据采样率计算，纵坐标显示相关系数
        plt.plot(corr)
        plt.xlabel('Time')
        #plt.xlim(0,len(corr)/samplerate)
        plt.ylabel('Correlation')
        plt.title('Cross-Correlation')
        plt.savefig(f'{file_path}/cross_correlation.png')
        plt.close()
        return FileResponse(path=f'{file_path}/cross_correlation.png',filename='cross_correlation.png')

        
@router.post("/correlation_cut")
async def correlation_cut(files: UploadFile = File(...),
                          filesRef: UploadFile = File(...),
                          max_correlation: float = 100):
    #filesref是参考文件，仅支持.wav文件，files是待处理文件，支持.wav和.zip
    #两者如果是双声道，默认只取左边声道
    #files使用extractor的规则
    corNum=max_correlation
    if filesRef.filename.endswith('.wav') and (files.filename.endswith('.wav') or files.filename.endswith('.zip')):
        file_extractor = FileExtractor(prefix="correlation_cut")
        file_path = file_extractor.extract(files)
        file_list = file_extractor.get_file_list('.wav')
        #读取参考文件
        data_ref, samplerate_ref = sf.read(filesRef.file)
        #如果是双声道文件，默认只取左边声道
        if data_ref.ndim == 2:
            data_ref = data_ref[:, 0]
        #参考文件保存到本地
        filesRef.file.seek(0)
        sf.write(f"{file_path}/reference_{str(uuid.uuid4())}.wav", data_ref, samplerate_ref)
        #读取待处理文件
        for file in file_list:
            data, samplerate = sf.read(file)
            #如果是双声道文件，默认只取左边声道
            if data.ndim == 2:
                data = data[:, 0]
            #计算相关系数
            #print(f"len(data)/samplerate:{len(data)/samplerate}")
            corr = signal.correlate(data, data_ref, mode='valid')   
            #plt.plot(corr)
            #plt.show()
            #遍历寻找相关系数大于max_correlation的点，如果有，则取范围内0-0.2s最大值的索引
            #取得所有最大值index后，按照index值和ref文件的长度裁剪音频
            #音频保存到file_path/cut_audio/文件夹下
            #压缩成压缩包，return
            #如果文件夹不存在，则创建文件夹
            os.makedirs(f"{file_path}/cut_audio",exist_ok=True)
            #遍历寻找相关系数大于max_correlation的点
            count=0
            skip_count=samplerate
            print(f"len(corr):{len(corr)/samplerate}")
            for i in range(len(corr)):
                if skip_count<=samplerate:
                    skip_count+=1
                    continue
                if corr[i] > corNum:
                    #取范围内0-0.2s最大值的索引
                    INDEX_RANGE=int(0.2*samplerate)
                    count+=1
                    #print(f"cor[i]:{corr[i]}")
                    #print(f"corNum:{corNum}")
                    max_index = np.argmax(corr[i:i+INDEX_RANGE])
                    #裁剪音频
                    #print(f"max_index:{max_index}")
                    #print(f"maxValue:{corr[max_index]}")
                    #print(f"butMaxValueIn02s:{corr[max_index+int(0.2*samplerate)]}")
                    
                    data_cut = data[max_index:max_index+len(data_ref)]
                    #保存音频，命名按照文件名+实际序号1，2，3等
                    sf.write(f"{file_path}/cut_audio/{file.name}_{count}.wav", data_cut, samplerate)
                    #print(f"max_index/samplerate:{max_index/samplerate}")

                    skip_count=0

            #压缩成压缩包，return
            with zipfile.ZipFile(f"{file_path}/cut_audio.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(f"{file_path}/cut_audio"):
                    for file in files:
                        zipf.write(os.path.join(root, file), os.path.join(root, file))
            return FileResponse(path=f"{file_path}/cut_audio.zip",filename='cut_audio.zip')

#响度均一化
@router.post("/equalize_loudness")
async def equalize_loudness(files: UploadFile = File(...),
                            weighting: str = "Total_RMS"):
    """
    对音频文件进行响度均一化处理
    
    Parameters:
    -----------
    files: UploadFile
        上传的zip文件，包含需要处理的音频文件
    weighting: str
        响度计算方式，可选"Total_RMS"或"A_weighting"，默认为"Total_RMS"
    
    Returns:
    --------
    FileResponse
        处理后的zip文件
    """
    print(f"weighting:{weighting}")
    # 使用FileExtractor处理文件
    file_extractor = FileExtractor(prefix="loudness_equalize")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    
    # 存储每个文件的响度值
    loudness_values = []
    
    # 计算每个文件的响度
    for file in file_list:
        data, samplerate = sf.read(file)
        
        if weighting == "Total_RMS":
            if data.ndim == 1:
                loudness = wave_analyzer.dB(np.sqrt(np.mean(data**2)))
            else:
                # 对于立体声，取左右声道的平均值
                loudness_L = wave_analyzer.dB(np.sqrt(np.mean(data[:,0]**2)))
                loudness_R = wave_analyzer.dB(np.sqrt(np.mean(data[:,1]**2)))
                loudness = (loudness_L + loudness_R) / 2
        else:  # A_weighting
            if data.ndim == 1:
                A_weighted = wave_analyzer.A_weight(data, samplerate)
                loudness = wave_analyzer.dB(wave_analyzer.rms_flat(A_weighted))
            else:
                A_weighted_L = wave_analyzer.A_weight(data[:,0], samplerate)
                A_weighted_R = wave_analyzer.A_weight(data[:,1], samplerate)
                loudness_L = wave_analyzer.dB(wave_analyzer.rms_flat(A_weighted_L))
                loudness_R = wave_analyzer.dB(wave_analyzer.rms_flat(A_weighted_R))
                loudness = (loudness_L + loudness_R) / 2
        
        loudness_values.append((file, loudness))
    
    # 找到最小响度值
    min_loudness = min(loudness for _, loudness in loudness_values)
    
    # 创建输出目录
    output_dir = os.path.join(file_path, "equalized")
    os.makedirs(output_dir, exist_ok=True)
    
    # 调整每个文件的响度
    for file, loudness in loudness_values:
        data, samplerate = sf.read(file)
        # 计算需要调整的增益
        gain_db = min_loudness - loudness
        gain_linear = wave_analyzer.db_to_linear_correct(gain_db)
        print(f"file:{file}")
        print(f"gain_db:{gain_db}")
        print(f"gain_linear:{gain_linear}")
        # 应用增益
        adjusted_data = data * gain_linear
        
        # 保存调整后的文件
        output_path = os.path.join(output_dir, os.path.basename(file))
        sf.write(output_path, adjusted_data, samplerate)
    
    # 创建zip文件
    zip_path = os.path.join(file_path, "equalized_files.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    
    return FileResponse(path=zip_path, filename="equalized_files.zip")

@router.post("/compareDiff")
async def compareDiff(files: UploadFile = File(...),
                     type: str = "deg",
                     compare_id: str = None):
    """
    比较两个音频文件的差异
    
    Args:
        files: 上传的音频文件
        type: 文件类型，"ref"表示参考文件，"deg"表示待比较文件
        compare_id: 用于标识一组比较的唯一ID
        
    Returns:
        JSONResponse: 包含比较结果的JSON响应
    """
    # 确保目录存在
    base_dir = "files/compareDiff"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # 处理参考文件
    if type == "ref":
        # 生成新的uuid
        if not compare_id:
            compare_id = str(uuid.uuid4())
        
        # 创建uuid对应的目录
        ref_dir = os.path.join(base_dir, compare_id)
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        
        # 保存参考文件
        ref_path = os.path.join(ref_dir, "ref.wav")
        with open(ref_path, "wb") as f:
            f.write(await files.read())
            
        # 保存高通滤波后的参考文件
        ref_data = load(ref_path)
        ref_filtered = highpass_filter(ref_data['signal'], 
                                     AudioCompareParams.HIGH_PASS_CUTOFF, 
                                     ref_data['fs'],
                                     AudioCompareParams.FILTER_ORDER)
        ref_highpass_path = os.path.join(ref_dir, "ref_highpass.wav")
        sf.write(ref_highpass_path, ref_filtered, ref_data['fs'])
        
        return JSONResponse(
            content={
                "message": "参考文件保存成功",
                "uuid": compare_id,
                "type": "ref",
                "answer": "NA"
            }
        )
    
    # 处理待比较文件
    elif type == "deg":
        if not compare_id:
            return JSONResponse(
                status_code=400,
                content={"error": "待比较文件需要提供uuid"}
            )
        
        # 检查参考文件是否存在
        ref_path = os.path.join(base_dir, compare_id, "ref.wav")
        if not os.path.exists(ref_path):
            return JSONResponse(
                content={
                    "message": "未找到对应的参考文件",
                    "uuid": compare_id,
                    "type": "deg",
                    "answer": "NoRef"
                }
            )
        
        # 保存待比较文件
        deg_dir = os.path.join(base_dir, compare_id)
        # 处理文件冲突
        base_path = os.path.join(deg_dir, "deg")
        counter = 1
        deg_path = f"{base_path}.wav"
        while os.path.exists(deg_path):
            deg_path = f"{base_path}_{counter}.wav"
            counter += 1
            
        with open(deg_path, "wb") as f:
            f.write(await files.read())
            
        # 保存高通滤波后的待比较文件
        deg_data = load(deg_path)
        deg_filtered = highpass_filter(deg_data['signal'], 
                                     AudioCompareParams.HIGH_PASS_CUTOFF, 
                                     deg_data['fs'],
                                     AudioCompareParams.FILTER_ORDER)
        deg_highpass_path = os.path.splitext(deg_path)[0] + "_highpass.wav"
        sf.write(deg_highpass_path, deg_filtered, deg_data['fs'])
        
        try:
            # 计算RMS差值
            diff_result = calculate_rms_difference(ref_path, deg_path)
            
            # 生成差值图
            plot_path = os.path.join(deg_dir, "diff_plot.png")
            channel_type = 'stereo' if diff_result['channels']['ref'] == 2 else 'mono'
            plot_rms_difference(diff_result, diff_result['fs'], plot_path, channel_type)
            
            # 判断结果
            answer = "PASS"
            if channel_type == 'mono':
                if diff_result['mono']['max_diff_db'] > AudioCompareParams.MAX_DIFF_DB_THRESHOLD:
                    answer = "FAIL"
            else:
                if diff_result['stereo']['left']['max_diff_db'] > AudioCompareParams.MAX_DIFF_DB_THRESHOLD or \
                   diff_result['stereo']['right']['max_diff_db'] > AudioCompareParams.MAX_DIFF_DB_THRESHOLD:
                    answer = "FAIL"
            
            # 准备返回结果
            response_content = {
                "message": "比较完成",
                "uuid": compare_id,
                "type": "deg",
                "channels": diff_result['channels'],
                "plot_path": plot_path,
                "answer": answer,
                "threshold": AudioCompareParams.MAX_DIFF_DB_THRESHOLD,
                "window_params": diff_result['window_params']
            }
            
            # 根据声道类型添加相应的差值数据
            if channel_type == 'mono':
                response_content.update({
                    "mean_diff": diff_result['mono']['mean_diff'],
                    "max_diff": diff_result['mono']['max_diff'],
                    "max_diff_db": diff_result['mono']['max_diff_db'],
                    "std_diff": diff_result['mono']['std_diff']
                })
            else:
                response_content.update({
                    "left_channel": {
                        "mean_diff": diff_result['stereo']['left']['mean_diff'],
                        "max_diff": diff_result['stereo']['left']['max_diff'],
                        "max_diff_db": diff_result['stereo']['left']['max_diff_db'],
                        "std_diff": diff_result['stereo']['left']['std_diff']
                    },
                    "right_channel": {
                        "mean_diff": diff_result['stereo']['right']['mean_diff'],
                        "max_diff": diff_result['stereo']['right']['max_diff'],
                        "max_diff_db": diff_result['stereo']['right']['max_diff_db'],
                        "std_diff": diff_result['stereo']['right']['std_diff']
                    }
                })
            
            return JSONResponse(content=response_content)
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "error": f"比较过程出错: {str(e)}",
                    "uuid": compare_id,
                    "type": "deg",
                    "answer": "ERROR"
                }
            )
    
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "无效的type参数", "answer": "ERROR"}
        )
