from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from utils.safe_extract import FileExtractor
import utils.waveform_analysis.wave_analyzer  as wave_analyzer
import soundfile as sf
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import uuid
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
        corr = np.correlate(data, data_ref, mode='valid')
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
            corr = np.correlate(data, data_ref, mode='valid')   
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
                    count+=1
                    #print(f"cor[i]:{corr[i]}")
                    #print(f"corNum:{corNum}")
                    max_index = np.argmax(corr[i:i+int(0.2*samplerate)])
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
