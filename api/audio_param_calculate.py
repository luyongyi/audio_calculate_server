from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from utils.safe_extract import FileExtractor
import utils.waveform_analysis.wave_analyzer  as wave_analyzer
import soundfile as sf
import csv
from utils.waveform_analysis.thd import THD

import numpy as np
from mosqito.sq_metrics import tnr_ecma_st
from mosqito.utils import load
from scipy.signal import resample
from mosqito.sq_metrics.loudness.loudness_zwst.loudness_zwst import loudness_zwst
from mosqito.sq_metrics.roughness.roughness_ecma.roughness_ecma import roughness_ecma
from mosqito.sq_metrics.sharpness.sharpness_din.sharpness_din_st import sharpness_din_st

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
                
                total_RMS_R = float('inf')
                A_weightRMS_R = float('inf')
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
                C80_R = float('inf')
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
    SPLin = wave_analyzer.db_to_linear(abs(st_calib)+abs(dBSPL))
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
            
            data, samplerate = load(str(file), wav_calib=SPLin)
            # 判断单声道还是立体声
            if data.ndim == 1:
                # 单声道处理
                t_tnr, tnr, prom, freq = tnr_ecma_st(data, samplerate, prominence=True)
                tonality_L = t_tnr[0]
                tonality_R = float('inf')
            else:
                # 立体声处理
                t_tnr_L, tnr_L, prom_L, freq_L = tnr_ecma_st(data[:,0], samplerate, prominence=True)
                t_tnr_R, tnr_R, prom_R, freq_R = tnr_ecma_st(data[:,1], samplerate, prominence=True)
                tonality_L = t_tnr_L[0]
                tonality_R = t_tnr_R[0]
            
            writer.writerow([file.name, tonality_L, tonality_R])
        
    return FileResponse(path=csvName, filename='audio_param.csv')

@router.post("/loudness")
async def get_loudness(files: UploadFile = File(...),
                       st_calib:float=94,
                       dBSPL:float=-25,
                       field_type: str = "free"):
    """
    计算音频文件的响度值
    
    Parameters:
    -----------
    files: UploadFile
        上传的音频文件
    st_calib: float
        校准值，默认为94
    dBSPL: float
        SPL值，默认为-25
    field_type: str
        声场类型，可选"free"（自由场）或"diffuse"（扩散场），默认为"free"
    
    Returns:
    --------
    FileResponse
        包含响度计算结果的CSV文件
    """
    SPLin = wave_analyzer.db_to_linear(abs(st_calib)+abs(dBSPL))
    file_extractor = FileExtractor(prefix="loudness")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    
    csvHeader = ['filename', 'loudness_L', 'loudness_R', 'field_type']
    csvName = f'{file_path}/audio_param.csv'
    
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        
        for file in file_list:
            data, samplerate = load(str(file), wav_calib=SPLin)
            
            # 确保采样率至少为48kHz
            if samplerate < 48000:
                data = resample(data, int(48000 * len(data) / samplerate))
                samplerate = 48000
            
            # 判断单声道还是立体声
            if data.ndim == 1:
                # 单声道处理
                N, N_specific, bark_axis = loudness_zwst(data, samplerate, field_type)
                loudness_L = round(N, 2)
                loudness_R = float('inf')
            else:
                # 立体声处理
                N_L, N_specific_L, bark_axis_L = loudness_zwst(data[:,0], samplerate, field_type)
                N_R, N_specific_R, bark_axis_R = loudness_zwst(data[:,1], samplerate, field_type)
                loudness_L = round(N_L, 2)
                loudness_R = round(N_R, 2)
            
            # 将结果写入CSV文件
            writer.writerow([file.name, loudness_L, loudness_R, field_type])
    
    return FileResponse(path=csvName, filename='audio_param.csv')

@router.post("/roughness")
async def get_roughness(files: UploadFile = File(...),
                       st_calib:float=94,
                       dBSPL:float=-25):
    """
    计算音频文件的粗糙度值
    
    Parameters:
    -----------
    files: UploadFile
        上传的音频文件
    st_calib: float
        校准值，默认为94
    dBSPL: float
        SPL值，默认为-25
    
    Returns:
    --------
    FileResponse
        包含粗糙度计算结果的CSV文件
    """
    SPLin = wave_analyzer.db_to_linear(abs(st_calib)+abs(dBSPL))
    file_extractor = FileExtractor(prefix="roughness")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    
    csvHeader = ['filename', 'roughness_L', 'roughness_R']
    csvName = f'{file_path}/audio_param.csv'
    
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        
        for file in file_list:
            data, samplerate = load(str(file), wav_calib=SPLin)
            
            # 确保采样率至少为48kHz
            if samplerate < 48000:
                data = resample(data, int(48000 * len(data) / samplerate))
                samplerate = 48000
            
            # 判断单声道还是立体声
            if data.ndim == 1:
                # 单声道处理
                R, R_time, R_spec, bark_axis, t_50 = roughness_ecma(data, samplerate)
                roughness_L = round(R, 2)
                roughness_R = float('inf')
            else:
                # 立体声处理
                R_L, R_time_L, R_spec_L, bark_axis_L, t_50_L = roughness_ecma(data[:,0], samplerate)
                R_R, R_time_R, R_spec_R, bark_axis_R, t_50_R = roughness_ecma(data[:,1], samplerate)
                roughness_L = round(R_L, 2)
                roughness_R = round(R_R, 2)
            
            # 将结果写入CSV文件
            writer.writerow([file.name, roughness_L, roughness_R])
    
    return FileResponse(path=csvName, filename='audio_param.csv')

@router.post("/sharpness")
async def get_sharpness(files: UploadFile = File(...),
                       st_calib:float=94,
                       dBSPL:float=-25,
                       weighting: str = "din",
                       field_type: str = "free"):
    """
    计算音频文件的锐度值
    
    Parameters:
    -----------
    files: UploadFile
        上传的音频文件
    st_calib: float
        校准值，默认为94
    dBSPL: float
        SPL值，默认为-25
    weighting: str
        锐度计算方法，可选"din"、"aures"、"bismarck"、"fastl"，默认为"din"
    field_type: str
        声场类型，可选"free"（自由场）或"diffuse"（扩散场），默认为"free"
    
    Returns:
    --------
    FileResponse
        包含锐度计算结果的CSV文件
    """
    SPLin = wave_analyzer.db_to_linear(abs(st_calib)+abs(dBSPL))
    file_extractor = FileExtractor(prefix="sharpness")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    
    csvHeader = ['filename', 'sharpness_L', 'sharpness_R', 'weighting', 'field_type']
    csvName = f'{file_path}/audio_param.csv'
    
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        
        for file in file_list:
            data, samplerate = load(str(file), wav_calib=SPLin)
            
            # 确保采样率至少为48kHz
            if samplerate < 48000:
                data = resample(data, int(48000 * len(data) / samplerate))
                samplerate = 48000
            
            # 判断单声道还是立体声
            if data.ndim == 1:
                # 单声道处理
                S = sharpness_din_st(data, samplerate, weighting=weighting, field_type=field_type)
                sharpness_L = round(S, 2)
                sharpness_R = float('inf')
            else:
                # 立体声处理
                S_L = sharpness_din_st(data[:,0], samplerate, weighting=weighting, field_type=field_type)
                S_R = sharpness_din_st(data[:,1], samplerate, weighting=weighting, field_type=field_type)
                sharpness_L = round(S_L, 2)
                sharpness_R = round(S_R, 2)
            
            # 将结果写入CSV文件
            writer.writerow([file.name, sharpness_L, sharpness_R, weighting, field_type])
    
    return FileResponse(path=csvName, filename='audio_param.csv')

@router.post("/THD")
async def get_THD(files: UploadFile = File(...)):
    """
    计算音频文件的THD值
    
    Parameters:
    -----------
    files: UploadFile
        上传的音频文件
    
    Returns:
    --------
    FileResponse
        包含THD计算结果的CSV文件
    """
    file_extractor = FileExtractor(prefix="THD")
    file_path = file_extractor.extract(files)
    file_list = file_extractor.get_file_list('.wav')
    csvHeader = ['filename', 'THD_L', 'THD_R']
    csvName = f'{file_path}/audio_param.csv'
    
    with open(csvName, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csvHeader)
        
        for file in file_list:
            data, samplerate = sf.read(file)
            # 判断单声道还是立体声
            if data.ndim == 1:
                THD_L = THD(data, samplerate)
                THD_R = float('inf')
            else:
                THD_L = THD(data[:,0], samplerate)
                THD_R = THD(data[:,1], samplerate)
            
            # 将结果写入CSV文件
            writer.writerow([file.name, THD_L, THD_R])
    
    return FileResponse(path=csvName, filename='audio_param.csv')


