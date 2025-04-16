from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from utils.safe_extract import FileExtractor
import utils.waveform_analysis.wave_analyzer  as wave_analyzer
import soundfile as sf
import csv
import numpy as np
from mosqito.sq_metrics import tnr_ecma_st
from mosqito.utils import load
router = APIRouter()

@router.post("/RMS")
async def get_RMS(files: UploadFile = File(...)):
    file_extractor = FileExtractor(prefix="RMS")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    csvHeader=['filename','total_RMS_L','total_RMS_R','A_weightRMS_L','A_weightRMS_R']
    csvName = f'{file_path}/audio_param.csv'
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        for file in file_list:
            data, samplerate = sf.read(file)
            #这里要判断data是单声道还是立体声
            #所有单位要用wave_analyzer.dB转换单位
            if data.ndim == 1:
                total_RMS_L = wave_analyzer.dB(np.sqrt(np.mean(data**2)))
                A_weightRMS_L = wave_analyzer.A_weight(data, samplerate)
                A_weightRMS_L = wave_analyzer.dB(wave_analyzer.rms_flat(A_weightRMS_L))
                
                total_RMS_R = "/"
                A_weightRMS_R = "/"
            else:
                total_RMS_L = np.sqrt(np.mean(data[:,0]**2))
                total_RMS_R = np.sqrt(np.mean(data[:,1]**2))
                total_RMS_L = wave_analyzer.dB(total_RMS_L)
                total_RMS_R = wave_analyzer.dB(total_RMS_R)
                A_weightRMS_L = wave_analyzer.A_weight(data[:,0], samplerate)
                A_weightRMS_R = wave_analyzer.A_weight(data[:,1], samplerate)
                A_weightRMS_L = wave_analyzer.dB(wave_analyzer.rms_flat(A_weightRMS_L))
                A_weightRMS_R = wave_analyzer.dB(wave_analyzer.rms_flat(A_weightRMS_R))

            # 将结果写入CSV文件,并且仅保留两位小数
            #writer.writerow([file.name,total_RMS_L,total_RMS_R,A_weightRMS_L,A_weightRMS_R])
            writer.writerow([file.name,round(total_RMS_L,2),round(total_RMS_R,2),round(A_weightRMS_L,2),round(A_weightRMS_R,2)])
        
    return FileResponse(path=csvName,filename='audio_param.csv')

@router.post("/C80")
async def get_C80(files: UploadFile = File(...)):
    file_extractor = FileExtractor(prefix="C80")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    csvHeader=['filename','C80_L','C80_R']
    csvName = f'{file_path}/audio_param.csv'
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        for file in file_list:
            data, samplerate = sf.read(file)
            #这里要判断data是单声道还是立体声
            if data.ndim == 1:
                C80_L = wave_analyzer.calculate_C80(data, samplerate)
                C80_R = "/"
            else:
                C80_L = wave_analyzer.calculate_C80(data[:,0], samplerate)
                C80_R = wave_analyzer.calculate_C80(data[:,1], samplerate)
                C80_L = round(C80_L,2)
                C80_R = round(C80_R,2)
            
            # 将结果写入CSV文件,并且仅保留两位小数
            writer.writerow([file.name, C80_L, C80_R])
        
    return FileResponse(path=csvName,filename='audio_param.csv')

@router.post("/tonality")
async def get_tonality(files: UploadFile = File(...),
                       st_calib:float=94,
                       dBSPL:float=-25):
    file_extractor = FileExtractor(prefix="tonality")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    csvHeader=['filename','tonality_L','tonality_R']
    csvName = f'{file_path}/audio_param.csv'
    
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        for file in file_list:
            # 使用 mosqito 的 load 方法加载音频文件
            
            data, samplerate = load(str(file), wav_calib=0.01)
            # 判断单声道还是立体声
            if data.ndim == 1:
                # 单声道处理
                t_tnr, tnr, prom, freq = tnr_ecma_st(data, samplerate, prominence=True)
                tonality_L = t_tnr[0]
                tonality_R = "/"
            else:
                # 立体声处理
                t_tnr_L, tnr_L, prom_L, freq_L = tnr_ecma_st(data[:,0], samplerate, prominence=True)
                t_tnr_R, tnr_R, prom_R, freq_R = tnr_ecma_st(data[:,1], samplerate, prominence=True)
                tonality_L = t_tnr_L[0]
                tonality_R = t_tnr_R[0]
            
            writer.writerow([file.name, tonality_L, tonality_R])
        
    return FileResponse(path=csvName, filename='audio_param.csv')
