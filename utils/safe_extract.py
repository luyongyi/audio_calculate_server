'''
safe_extract.py
主要用于安全解压文件和移动文件
主要处理几个类型：
1、如果文件是zip文件，则创建files目录的子目录（uuid4），然后解压文件到该目录，并返回解压后的文件路径
2、如果文件是wav文件，则创建files目录的子目录（uuid4），然后移动文件到该目录，并返回移动后的文件路径
3、如果文件是其他类型，则直接返回None以让上层处理非法投递
'''

import os
import shutil
import uuid
from pathlib import Path
from typing import Union
import zipfile
from fastapi import UploadFile

class FileExtractor:
    def __init__(self, base_dir: Union[str, Path] = "files",prefix:str = ""):
        """
        初始化文件提取器
        
        Args:
            base_dir: 基础目录，默认为"files"
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.prefix = prefix
    
    def extract(self, upload_file: UploadFile) -> Path:
        """
        处理上传的文件
        
        Args:
            upload_file: FastAPI的UploadFile对象
            
        Returns:
            Path: 处理后的文件或目录路径
            
        Raises:
            ValueError: 如果文件处理失败
        """
        # 创建唯一子目录
        unique_dir = self.base_dir / (self.prefix + "_" + str(uuid.uuid4()))
        unique_dir.mkdir()
        
        # 获取文件扩展名
        file_ext = Path(upload_file.filename).suffix.lower()
        self.unique_dir = unique_dir
        # 保存文件到临时路径
        temp_path = unique_dir / upload_file.filename
        
        try:
            # 保存文件
            with open(temp_path, 'wb') as out_file:
                content = upload_file.file.read()
                out_file.write(content)
            
            # 根据文件类型处理
            if file_ext == '.zip':
                # 解压zip文件
                with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                    zip_ref.extractall(unique_dir)
                # 删除原始zip文件
                temp_path.unlink()
                return unique_dir
            elif file_ext == '.wav':
                # wav文件已经移动到正确位置，直接返回路径
                return unique_dir
            else:
                # 其他文件类型，返回保存后的路径
                return None
                
        except Exception as e:
            # 如果处理失败，清理临时文件
            if temp_path.exists():
                temp_path.unlink()
            if unique_dir.exists():
                shutil.rmtree(unique_dir)
            raise ValueError(f"文件处理失败: {str(e)}")
    def get_file_list(self,extention:str) -> list:
        #这里获取unique_dir下的所有文件
        return list(self.unique_dir.glob(f'*{extention}'))
    def cleanup(self) -> None:
        """
        自动清理处理后的文件或目录
        """
        path = self.unique_dir
        path = Path(path)
        if path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)