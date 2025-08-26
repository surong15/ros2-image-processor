#!/usr/bin/env python3
# milvus一直斷線，好像比之前嚴重
# web查看器不會根據時間順序排序

"""
ROS2 Image Processor GUI Application
帶有圖形界面的 ROS2 影像處理應用程式
可手動控制存儲、監控訂閱狀態和資料庫狀態
"""

import sys
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import queue
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import json
import uuid
import base64
from io import BytesIO
import requests
import traceback
import argparse
import yaml
import webbrowser
from abc import ABC, abstractmethod
import socket
import subprocess
import http.server
import socketserver

# NumPy 兼容性檢查
def check_numpy_compatibility():
    try:
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        if major_version >= 2:
            print(f"⚠️ NumPy {numpy_version} detected. Please downgrade: pip install 'numpy<2.0'")
            return False
        return True
    except ImportError:
        return False

if not check_numpy_compatibility():
    sys.exit(1)

import numpy as np

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as ROS_Image
    from cv_bridge import CvBridge
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    ROS2_AVAILABLE = True
except ImportError as e:
    ROS2_AVAILABLE = False
    print(f"❌ ROS2 not available: {e}")

# Milvus imports
try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

# Qdrant imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# PIL import
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

@dataclass
class CameraDataPacket:
    """統一的相機資料包"""
    image: np.ndarray
    timestamp: datetime
    position: list[float]
    rotation: list[float]
    coordinate_frame: str
    coordinate_method: str
    capture_time_iso: str
    frame_id: int

class VectorDBManager(ABC):
    """向量資料庫管理器抽象介面"""
    
    @abstractmethod
    def connect(self) -> bool:
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        pass
    
    @abstractmethod
    def store_data_packet(self, data_packet: CameraDataPacket, ai_answer: str) -> bool:
        pass
    
    @abstractmethod
    def get_total_count(self) -> int:
        pass
    
    @abstractmethod
    def get_db_info(self) -> str:
        pass
    
    @abstractmethod
    def update_config(self, new_config: dict):
        pass

class DatabaseManagerFactory:
    """資料庫管理器工廠"""
    
    @staticmethod
    def create_manager(db_type: str, config: dict, status_queue: queue.Queue) -> VectorDBManager:
        if db_type == 'milvus':
            return MilvusManager(config, status_queue)
        elif db_type == 'qdrant':
            return QdrantManager(config, status_queue)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

class StatusManager:
    """狀態管理器"""
    def __init__(self):
        self.ros2_connected = False
        self.image_received = False
        self.tf_received = False
        self.milvus_connected = False
        self.qdrant_connected = False
        self.ollama_connected = False
        self.storage_active = False
        
        self.image_count = 0
        self.tf_count = 0
        self.stored_count = 0
        self.error_count = 0
        
        self.last_image_time = None
        self.last_tf_time = None
        self.last_storage_time = None

class ROS2CameraSubscriber(Node):
    """ROS2 相機影像訂閱器"""
    
    def __init__(self, topic_name: str, status_queue: queue.Queue):
        super().__init__('gui_image_subscriber')
        self.topic_name = topic_name
        self.status_queue = status_queue
        self.latest_image = None
        self.latest_timestamp = None
        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.frame_count = 0
        
        # 創建訂閱器
        self.subscription = self.create_subscription(
            ROS_Image,
            topic_name,
            self.image_callback,
            10
        )
        
        self.status_queue.put(('log', f"✅ Image subscriber created: {topic_name}"))
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            
            with self.image_lock:
                self.latest_image = cv_image
                self.latest_timestamp = datetime.now()
                self.frame_count += 1
            
            # Terminal 輸出（減少頻率）
            if self.frame_count <= 3 or self.frame_count % 50 == 0:
                print(f"📷 [ROS2] Received image #{self.frame_count}, shape: {cv_image.shape}")
            
            # 更新狀態
            self.status_queue.put(('image_received', {
                'count': self.frame_count,
                'timestamp': self.latest_timestamp,
                'shape': cv_image.shape
            }))
            
        except Exception as e:
            print(f"❌ [ROS2] Image callback error: {e}")
            self.status_queue.put(('error', f"Image callback error: {e}"))
    
    def get_latest_image(self):
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.copy(), self.latest_timestamp
            return None, None

class ROS2TFSubscriber(Node):
    """ROS2 TF 變換訂閱器"""
    
    def __init__(self, tf_topic: str, target_frame: str, status_queue: queue.Queue):
        super().__init__('gui_tf_subscriber')
        self.tf_topic = tf_topic
        self.target_frame = target_frame
        self.status_queue = status_queue
        self.latest_transform = None
        self.latest_timestamp = None
        self.tf_lock = threading.Lock()
        self.tf_count = 0
        self.found_target = False
        
        # 創建 TF 訂閱器
        self.subscription = self.create_subscription(
            TFMessage,
            tf_topic,
            self.tf_callback,
            10
        )
        
        self.status_queue.put(('log', f"✅ TF subscriber created: {tf_topic}, target: {target_frame}"))
    
    def tf_callback(self, msg):
        try:
            self.tf_count += 1
            system_timestamp = datetime.now()
            
            # 尋找目標 frame
            found = False
            for transform in msg.transforms:
                if transform.child_frame_id == self.target_frame:
                    with self.tf_lock:
                        self.latest_transform = transform
                        self.latest_timestamp = system_timestamp
                    found = True
                    if not self.found_target:
                        self.found_target = True
                        print(f"📡 [ROS2] ✅ Found target frame: {self.target_frame}")
                    break
            
            # Terminal 輸出（減少頻率）
            if self.tf_count <= 3:
                available_frames = [t.child_frame_id for t in msg.transforms]
                print(f"📡 [ROS2] TF message #{self.tf_count} with {len(msg.transforms)} transforms")
                print(f"📡 [ROS2] Available frames: {available_frames[:5]}...")
                print(f"📡 [ROS2] Looking for target: '{self.target_frame}' - {'✅ Found' if found else '❌ Not found'}")
            elif self.tf_count % 100 == 0:
                print(f"📡 [ROS2] TF messages received: {self.tf_count} (target: {'✅' if self.found_target else '❌'})")
            
            # 更新狀態
            self.status_queue.put(('tf_received', {
                'count': self.tf_count,
                'timestamp': system_timestamp,
                'found_target': found,
                'available_frames': [t.child_frame_id for t in msg.transforms][:5]
            }))
            
        except Exception as e:
            print(f"❌ [ROS2] TF callback error: {e}")
            self.status_queue.put(('error', f"TF callback error: {e}"))
    
    def get_latest_transform(self):
        with self.tf_lock:
            if self.latest_transform is not None:
                return self.latest_transform, self.latest_timestamp
            return None, None
    
    def extract_coordinates(self, transform_msg):
        try:
            translation = transform_msg.transform.translation
            rotation = transform_msg.transform.rotation
            
            position = [float(translation.x), float(translation.y), float(translation.z)]
            rotation_quat = [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]
            
            return {
                'position': position,
                'rotation': rotation_quat,
                'coordinate_frame': 'ros2_tf',
                'method': 'ros2_tf_transform'
            }
        except Exception as e:
            return {
                'position': [0.0, 0.0, 0.0],
                'rotation': [1.0, 0.0, 0.0, 0.0],
                'coordinate_frame': 'error',
                'method': 'error'
            }

class OllamaVLMAnalyzer:
    """Ollama VLM 分析器"""
    
    def __init__(self, config: dict, status_queue: queue.Queue):
        self.config = config
        self.status_queue = status_queue
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
        self.ollama_model = config.get('ollama_model', 'llava:latest')
        self.ai_timeout = config.get('ai_timeout', 30)
        self.enabled = config.get('enable_ai_analysis', True)
        self.connected = False
        
        # 檢查連接
        self._check_connection()
    
    def _check_connection(self):
        if not self.enabled:
            return
            
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                
                if any(self.ollama_model in name for name in model_names):
                    self.connected = True
                    self.status_queue.put(('ollama_connected', True))
                    self.status_queue.put(('log', f"✅ Ollama connected: {self.ollama_model}"))
                else:
                    self.connected = False
                    self.status_queue.put(('ollama_connected', False))
                    self.status_queue.put(('log', f"❌ Model {self.ollama_model} not found"))
            else:
                raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            self.connected = False
            self.status_queue.put(('ollama_connected', False))
            self.status_queue.put(('log', f"❌ Ollama connection failed: {e}"))
    
    def analyze_image(self, image_np):
        """分析圖像（加強錯誤處理）"""
        if not self.enabled or not self.connected:
            return "AI analysis disabled or not connected"
        
        try:
            if not PIL_AVAILABLE:
                return "PIL not available"
            
            print(f"🤖 [AI Analysis] Starting image analysis with {self.ollama_model}...")
            
            # 準備圖像
            pil_image = Image.fromarray(image_np.astype(np.uint8))
            
            # 縮放
            original_size = pil_image.size
            if max(pil_image.size) > 512:
                ratio = 512 / max(pil_image.size)
                new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"🤖 [AI Analysis] Image resized from {original_size} to {new_size}")
            
            # 轉換為 base64
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            print(f"🤖 [AI Analysis] Image encoded to base64 ({len(img_base64)} chars)")
            
            # 調用 Ollama API
            payload = {
                "model": self.ollama_model,
                "prompt": "請描述這張圖片，並用中文回答。",
                "images": [img_base64],
                "stream": False,
                "options": {"temperature": 0.3}
            }
            
            print(f"🤖 [AI Analysis] Sending request to {self.ollama_url}...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.ai_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "No description available")
                
                # 在 terminal 顯示 AI 分析結果
                print("🤖 " + "="*60)
                print("🤖 [AI Analysis] LLaVA 圖像描述結果:")
                print("🤖 " + "-"*60)
                print(f"🤖 {answer}")
                print("🤖 " + "="*60)
                
                return answer
            else:
                error_msg = f"AI analysis failed (HTTP {response.status_code})"
                print(f"🤖 [AI Analysis] ❌ {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"AI analysis error: {str(e)}"
            print(f"🤖 [AI Analysis] ❌ {error_msg}")
            self.status_queue.put(('error', f"Ollama analysis error: {e}"))
            return error_msg

class MilvusManager(VectorDBManager):
    """Milvus 資料庫管理器"""
    
    def __init__(self, config: dict, status_queue: queue.Queue):
        self.config = config
        self.status_queue = status_queue
        self.host = config.get('milvus_host', 'localhost')
        self.port = config.get('milvus_port', '19530')
        self.collection_name = config.get('collection_name', 'ros2_camera_images')
        self.vector_dim = config.get('vector_dim', 512)
        self.collection = None
        self.connected = False
        self.stored_count = 0
        
        # 不自動連接，等待手動連接
    
    def connect(self):
        """手動連接 Milvus"""
        return self._init_connection()
    
    def disconnect(self):
        """手動斷開 Milvus 連接"""
        try:
            if self.connected:
                if self.collection:
                    self.collection.release()
                    self.collection = None
                
                connections.disconnect("default")
                self.connected = False
                self.status_queue.put(('milvus_connected', False))
                self.status_queue.put(('log', "🔌 Milvus 連接已斷開"))
                return True
            return False
        except Exception as e:
            self.status_queue.put(('log', f"❌ Milvus 斷開失敗: {e}"))
            return False
    
    def update_config(self, new_config):
        """更新配置"""
        self.config.update(new_config)
        self.host = self.config.get('milvus_host', 'localhost')
        self.port = self.config.get('milvus_port', '19530')
        self.collection_name = self.config.get('collection_name', 'ros2_camera_images')
        self.vector_dim = self.config.get('vector_dim', 512)
    
    def _init_connection(self):
        """初始化 Milvus 連接"""
        if not MILVUS_AVAILABLE:
            self.status_queue.put(('milvus_connected', False))
            self.status_queue.put(('log', "❌ Milvus modules not available"))
            return False
        
        try:
            self.status_queue.put(('log', f"🔗 正在連接 Milvus {self.host}:{self.port}..."))
            
            # 先斷開現有連接
            try:
                connections.disconnect("default")
            except:
                pass
            
            # 連接 Milvus
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            
            # 創建或載入集合
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.status_queue.put(('log', f"✅ 載入現有集合: {self.collection_name}"))
            else:
                self._create_collection()
            
            if self.collection:
                self.collection.load()
                self.stored_count = self.collection.num_entities
                self.connected = True
                self.status_queue.put(('milvus_connected', True))
                self.status_queue.put(('log', f"✅ Milvus 連接成功: {self.stored_count} 筆現有記錄"))
                return True
            else:
                self.connected = False
                self.status_queue.put(('milvus_connected', False))
                self.status_queue.put(('log', "❌ 集合創建失敗"))
                return False
            
        except Exception as e:
            self.connected = False
            self.status_queue.put(('milvus_connected', False))
            self.status_queue.put(('log', f"❌ Milvus 連接失敗: {e}"))
            return False
    
    def _create_collection(self):
        """創建 Milvus 集合"""
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="ros2_topic", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="image_size", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="image_base64", dtype=DataType.VARCHAR, max_length=65000),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="ai_question", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="ai_answer", dtype=DataType.VARCHAR, max_length=3000),
                FieldSchema(name="ai_analysis_success", dtype=DataType.BOOL),
                FieldSchema(name="position_x", dtype=DataType.DOUBLE),
                FieldSchema(name="position_y", dtype=DataType.DOUBLE),
                FieldSchema(name="position_z", dtype=DataType.DOUBLE),
                FieldSchema(name="rotation_x", dtype=DataType.DOUBLE),
                FieldSchema(name="rotation_y", dtype=DataType.DOUBLE),
                FieldSchema(name="rotation_z", dtype=DataType.DOUBLE),
                FieldSchema(name="rotation_w", dtype=DataType.DOUBLE),
                FieldSchema(name="coordinate_frame", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="tf_topic", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="target_frame", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="capture_method", dtype=DataType.VARCHAR, max_length=50)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="ROS2 GUI Application images with coordinates and AI analysis"
            )
            
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            # 創建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            self.collection.create_index(field_name="image_vector", index_params=index_params)
            
            self.status_queue.put(('log', f"✅ Created new collection: {self.collection_name}"))
            
        except Exception as e:
            self.status_queue.put(('log', f"❌ Collection creation failed: {e}"))
    
    def store_data_packet(self, data_packet: CameraDataPacket, ai_answer: str) -> bool:
        """存儲資料包到 Milvus"""
        if not self.connected or not self.collection:
            return False
        
        try:
            print(f"💾 [Milvus] Starting storage for frame_{data_packet.frame_id}...")
            
            # 生成向量
            image_vector = self._image_to_vector(data_packet.image)
            print(f"💾 [Milvus] Generated feature vector (dim: {len(image_vector)})")
            
            # 壓縮圖像
            image_base64 = self._compress_image(data_packet.image)
            if not image_base64:
                print(f"❌ [Milvus] Image compression failed for frame_{data_packet.frame_id}")
                return False
            print(f"💾 [Milvus] Image compressed to {len(image_base64)} chars")
            
            # 準備數據
            ai_success = not ai_answer.startswith(("AI analysis", "No description"))
            image_size = f"{data_packet.image.shape[1]}x{data_packet.image.shape[0]}"
            
            metadata = json.dumps({
                "capture_method": "ros2_gui_application",
                "coordinate_method": data_packet.coordinate_method,
                "coordinate_frame": data_packet.coordinate_frame,
                "image_shape": list(data_packet.image.shape),
                "frame_id": data_packet.frame_id,
                "app_version": "gui_v1.0"
            })
            
            data = [{
                "id": str(uuid.uuid4()),
                "image_vector": image_vector.tolist(),
                "timestamp": data_packet.capture_time_iso,
                "ros2_topic": self.config.get('image_topic', 'unknown'),
                "image_size": image_size,
                "image_base64": image_base64,
                "metadata": metadata,
                "ai_question": "請描述這張圖片",
                "ai_answer": ai_answer,
                "ai_analysis_success": ai_success,
                "position_x": data_packet.position[0],
                "position_y": data_packet.position[1],
                "position_z": data_packet.position[2],
                "rotation_x": data_packet.rotation[1],
                "rotation_y": data_packet.rotation[2],
                "rotation_z": data_packet.rotation[3],
                "rotation_w": data_packet.rotation[0],
                "coordinate_frame": data_packet.coordinate_frame,
                "tf_topic": self.config.get('tf_topic', 'unknown'),
                "target_frame": self.config.get('target_frame', 'unknown'),
                "date": data_packet.timestamp.strftime("%Y-%m-%d"),
                "time": data_packet.timestamp.strftime("%H:%M:%S"),
                "capture_method": "ros2_gui_application"
            }]
            
            # 插入數據
            self.collection.insert(data)
            self.stored_count += 1
            
            # Terminal 輸出存儲結果
            print("💾 " + "="*50)
            print(f"💾 [Milvus] ✅ Successfully stored frame_{data_packet.frame_id}")
            print(f"💾 [Milvus] Position: [{data_packet.position[0]:.3f}, {data_packet.position[1]:.3f}, {data_packet.position[2]:.3f}]")
            print(f"💾 [Milvus] AI Analysis: {'✅ Success' if ai_success else '❌ Failed'}")
            print(f"💾 [Milvus] Total stored: {self.stored_count}")
            print("💾 " + "="*50)
            
            # 更新狀態
            self.status_queue.put(('data_stored', {
                'count': self.stored_count,
                'frame_id': data_packet.frame_id,
                'timestamp': data_packet.timestamp
            }))
            
            return True
            
        except Exception as e:
            print(f"❌ [Milvus] Storage error for frame_{data_packet.frame_id}: {e}")
            self.status_queue.put(('error', f"Milvus storage error: {e}"))
            return False
    
    def _image_to_vector(self, image: np.ndarray) -> np.ndarray:
        """將圖像轉換為特徵向量（簡化版）"""
        try:
            # 調整大小
            target_size = (224, 224)
            if PIL_AVAILABLE:
                pil_image = Image.fromarray(image.astype(np.uint8))
                resized = pil_image.resize(target_size)
                img_array = np.array(resized)
            else:
                h, w = image.shape[:2]
                target_h, target_w = target_size
                img_array = image[::h//target_h, ::w//target_w]
            
            # 簡單特徵提取
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
            
            # 統計特徵
            stats = [
                np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                np.median(gray)
            ]
            
            # 組合特徵
            features = np.concatenate([hist, stats])
            
            # 調整到目標維度
            if len(features) > self.vector_dim:
                features = features[:self.vector_dim]
            elif len(features) < self.vector_dim:
                padding = self.vector_dim - len(features)
                features = np.concatenate([features, np.zeros(padding)])
            
            # 正規化
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.astype(np.float32)
            
        except Exception as e:
            return np.random.rand(self.vector_dim).astype(np.float32)
    
    def _compress_image(self, image: np.ndarray, quality=75) -> str:
        """壓縮圖像為 base64"""
        try:
            if PIL_AVAILABLE:
                pil_image = Image.fromarray(image.astype(np.uint8))
                
                # 縮放
                if max(pil_image.size) > 800:
                    ratio = 800 / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # 壓縮
                buffer = BytesIO()
                pil_image.save(buffer, format='JPEG', quality=quality)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                return ""
        except Exception:
            return ""
    
    def get_total_count(self):
        """獲取總數量"""
        try:
            if self.collection:
                return self.collection.num_entities
        except:
            pass
        return self.stored_count
    
    def get_db_info(self) -> str:
        """獲取資料庫資訊"""
        if self.connected:
            return f"Milvus {self.host}:{self.port} | {self.collection_name} | {self.stored_count} records"
        else:
            return f"Milvus {self.host}:{self.port} | 未連接"

class QdrantManager(VectorDBManager):
    """Qdrant 資料庫管理器"""
    
    def __init__(self, config: dict, status_queue: queue.Queue):
        self.config = config
        self.status_queue = status_queue
        self.host = config.get('qdrant_host', 'localhost')
        self.port = config.get('qdrant_port', '6333')
        self.collection_name = config.get('qdrant_collection_name', 'ros2_camera_images')
        self.vector_dim = config.get('vector_dim', 512)
        self.client = None
        self.connected = False
        self.stored_count = 0
        
        # 不自動連接，等待手動連接
    
    def connect(self) -> bool:
        """手動連接 Qdrant"""
        return self._init_connection()
    
    def disconnect(self) -> bool:
        """手動斷開 Qdrant 連接"""
        try:
            if self.connected:
                self.client = None
                self.connected = False
                self.status_queue.put(('qdrant_connected', False))
                self.status_queue.put(('log', "🔌 Qdrant 連接已斷開"))
                return True
            return False
        except Exception as e:
            self.status_queue.put(('log', f"❌ Qdrant 斷開失敗: {e}"))
            return False
    
    def update_config(self, new_config: dict):
        """更新配置"""
        self.config.update(new_config)
        self.host = self.config.get('qdrant_host', 'localhost')
        self.port = self.config.get('qdrant_port', '6333')
        self.collection_name = self.config.get('qdrant_collection_name', 'ros2_camera_images')
        self.vector_dim = self.config.get('vector_dim', 512)
    
    def _init_connection(self) -> bool:
        """初始化 Qdrant 連接"""
        if not QDRANT_AVAILABLE:
            self.status_queue.put(('qdrant_connected', False))
            self.status_queue.put(('log', "❌ Qdrant modules not available"))
            return False
        
        try:
            self.status_queue.put(('log', f"🔗 正在連接 Qdrant {self.host}:{self.port}..."))
            
            # 連接 Qdrant
            self.client = QdrantClient(host=self.host, port=int(self.port))
            
            # 檢查集合是否存在，不存在則創建
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self._create_collection()
            
            # 獲取現有記錄數
            collection_info = self.client.get_collection(self.collection_name)
            self.stored_count = collection_info.points_count
            
            self.connected = True
            self.status_queue.put(('qdrant_connected', True))
            self.status_queue.put(('log', f"✅ Qdrant 連接成功: {self.stored_count} 筆現有記錄"))
            return True
            
        except Exception as e:
            self.connected = False
            self.status_queue.put(('qdrant_connected', False))
            self.status_queue.put(('log', f"❌ Qdrant 連接失敗: {e}"))
            return False
    
    def _create_collection(self):
        """創建 Qdrant 集合"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=Distance.COSINE
                )
            )
            self.status_queue.put(('log', f"✅ Created new Qdrant collection: {self.collection_name}"))
            
        except Exception as e:
            self.status_queue.put(('log', f"❌ Qdrant collection creation failed: {e}"))
    
    def store_data_packet(self, data_packet: CameraDataPacket, ai_answer: str) -> bool:
        """存儲資料包到 Qdrant"""
        if not self.connected or not self.client:
            return False
        
        try:
            print(f"💾 [Qdrant] Starting storage for frame_{data_packet.frame_id}...")
            
            # 生成向量
            image_vector = self._image_to_vector(data_packet.image)
            print(f"💾 [Qdrant] Generated feature vector (dim: {len(image_vector)})")
            
            # 壓縮圖像
            image_base64 = self._compress_image(data_packet.image)
            if not image_base64:
                print(f"❌ [Qdrant] Image compression failed for frame_{data_packet.frame_id}")
                return False
            print(f"💾 [Qdrant] Image compressed to {len(image_base64)} chars")
            
            # 準備數據
            ai_success = not ai_answer.startswith(("AI analysis", "No description"))
            image_size = f"{data_packet.image.shape[1]}x{data_packet.image.shape[0]}"
            
            # 創建點結構
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=image_vector.tolist(),
                payload={
                    "timestamp": data_packet.capture_time_iso,
                    "ros2_topic": self.config.get('image_topic', 'unknown'),
                    "image_size": image_size,
                    "image_base64": image_base64,
                    "ai_question": "請描述這張圖片",
                    "ai_answer": ai_answer,
                    "ai_analysis_success": ai_success,
                    "position_x": data_packet.position[0],
                    "position_y": data_packet.position[1],
                    "position_z": data_packet.position[2],
                    "rotation_x": data_packet.rotation[1],
                    "rotation_y": data_packet.rotation[2],
                    "rotation_z": data_packet.rotation[3],
                    "rotation_w": data_packet.rotation[0],
                    "coordinate_frame": data_packet.coordinate_frame,
                    "tf_topic": self.config.get('tf_topic', 'unknown'),
                    "target_frame": self.config.get('target_frame', 'unknown'),
                    "date": data_packet.timestamp.strftime("%Y-%m-%d"),
                    "time": data_packet.timestamp.strftime("%H:%M:%S"),
                    "capture_method": "ros2_gui_application",
                    "frame_id": data_packet.frame_id,
                    "app_version": "gui_v1.0"
                }
            )
            
            # 插入數據
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            self.stored_count += 1
            
            # Terminal 輸出存儲結果
            print("💾 " + "="*50)
            print(f"💾 [Qdrant] ✅ Successfully stored frame_{data_packet.frame_id}")
            print(f"💾 [Qdrant] Position: [{data_packet.position[0]:.3f}, {data_packet.position[1]:.3f}, {data_packet.position[2]:.3f}]")
            print(f"💾 [Qdrant] AI Analysis: {'✅ Success' if ai_success else '❌ Failed'}")
            print(f"💾 [Qdrant] Total stored: {self.stored_count}")
            print("💾 " + "="*50)
            
            # 更新狀態
            self.status_queue.put(('data_stored', {
                'count': self.stored_count,
                'frame_id': data_packet.frame_id,
                'timestamp': data_packet.timestamp
            }))
            
            return True
            
        except Exception as e:
            print(f"❌ [Qdrant] Storage error for frame_{data_packet.frame_id}: {e}")
            self.status_queue.put(('error', f"Qdrant storage error: {e}"))
            return False
    
    def _image_to_vector(self, image: np.ndarray) -> np.ndarray:
        """將圖像轉換為特徵向量（簡化版）"""
        try:
            # 調整大小
            target_size = (224, 224)
            if PIL_AVAILABLE:
                pil_image = Image.fromarray(image.astype(np.uint8))
                resized = pil_image.resize(target_size)
                img_array = np.array(resized)
            else:
                h, w = image.shape[:2]
                target_h, target_w = target_size
                img_array = image[::h//target_h, ::w//target_w]
            
            # 簡單特徵提取
            gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            hist, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
            
            # 統計特徵
            stats = [
                np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                np.median(gray)
            ]
            
            # 組合特徵
            features = np.concatenate([hist, stats])
            
            # 調整到目標維度
            if len(features) > self.vector_dim:
                features = features[:self.vector_dim]
            elif len(features) < self.vector_dim:
                padding = self.vector_dim - len(features)
                features = np.concatenate([features, np.zeros(padding)])
            
            # 正規化
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features.astype(np.float32)
            
        except Exception as e:
            return np.random.rand(self.vector_dim).astype(np.float32)
    
    def _compress_image(self, image: np.ndarray, quality=75) -> str:
        """壓縮圖像為 base64"""
        try:
            if PIL_AVAILABLE:
                pil_image = Image.fromarray(image.astype(np.uint8))
                
                # 縮放
                if max(pil_image.size) > 800:
                    ratio = 800 / max(pil_image.size)
                    new_size = tuple(int(dim * ratio) for dim in pil_image.size)
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                
                # 壓縮
                buffer = BytesIO()
                pil_image.save(buffer, format='JPEG', quality=quality)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            else:
                return ""
        except Exception:
            return ""
    
    def get_total_count(self) -> int:
        """獲取總數量"""
        try:
            if self.client and self.connected:
                collection_info = self.client.get_collection(self.collection_name)
                return collection_info.points_count
        except:
            pass
        return self.stored_count
    
    def get_db_info(self) -> str:
        """獲取資料庫資訊"""
        if self.connected:
            return f"Qdrant {self.host}:{self.port} | {self.collection_name} | {self.stored_count} records"
        else:
            return f"Qdrant {self.host}:{self.port} | 未連接"

class ROS2ImageProcessorGUI:
    """ROS2 圖像處理器 GUI 應用程式"""
    
    def __init__(self, master):
        print("🎛️ [GUI] Initializing ROS2 Image Processor GUI...")
        
        self.master = master
        self.master.title("ROS2 Image Processor GUI Application")
        self.master.geometry("1000x700")
        
        print("🎛️ [GUI] Setting up status management...")
        # 狀態管理
        self.status_queue = queue.Queue()
        self.status_manager = StatusManager()
        
        # 配置
        self.config = self._load_default_config()
        print(f"🎛️ [GUI] Configuration loaded: {self.config['image_topic']}, {self.config['tf_topic']}")
        
        # ROS2 組件
        self.image_subscriber = None
        self.tf_subscriber = None
        self.vlm_analyzer = None
        
        # 資料庫管理器字典
        self.db_managers = {}  # {'milvus': MilvusManager, 'qdrant': QdrantManager}
        self.current_db_type = self.config.get('database_type', 'milvus')
        
        # 控制狀態
        self.ros2_running = False
        self.storage_running = False
        self.current_data_packet = None
        self.data_lock = threading.Lock()
        
        # Web服务器状态
        self.web_server_running = False
        self.web_server_process = None
        self.web_server_port = self.config.get('web_viewer_port', 8889)
        
        print("🎛️ [GUI] Creating GUI components...")
        # 創建 GUI
        try:
            self._create_gui()
            print("✅ [GUI] GUI creation successful")
        except Exception as e:
            print(f"❌ [GUI] GUI creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        print("🎛️ [GUI] Starting status update loop...")
        # 啟動狀態更新循環
        try:
            self._start_status_update_loop()
            print("✅ [GUI] Status update loop started")
        except Exception as e:
            print(f"❌ [GUI] Status update loop failed: {e}")
        
        print("🎛️ [GUI] Initializing status display...")
        # 初始化狀態顯示
        try:
            self._update_db_control_ui()  # 更新資料庫控制UI
            self._update_current_db_status()
            print("✅ [GUI] Database status updated")
        except Exception as e:
            print(f"❌ [GUI] Database status update failed: {e}")
            
        try:
            self._update_ollama_info()
            print("✅ [GUI] Ollama info updated")
        except Exception as e:
            print(f"❌ [GUI] Ollama info update failed: {e}")
        
        print("✅ [GUI] GUI initialization completed")
        print("💡 [GUI] Ready for user interaction - all logs will appear in terminal")
    
    def _load_default_config(self):
        """載入預設配置"""
        return {
            # 資料庫選擇
            'database_type': 'milvus',  # 'milvus' 或 'qdrant'
            'allow_runtime_db_selection': True,  # 是否允許執行時選擇資料庫
            
            # ROS2 配置
            'image_topic': '/baymax/camera/image_raw',
            'tf_topic': '/baymax/tf',
            'target_frame': 'tn__7R05D00002_only_bottom_sim_',
            
            # Ollama 配置
            'ollama_url': 'http://localhost:11434',
            'ollama_model': 'llava:latest',
            'ai_timeout': 30,
            'enable_ai_analysis': True,
            
            # Milvus 配置
            'milvus_host': 'localhost',
            'milvus_port': '19530',
            'collection_name': 'ros2_camera_images',
            
            # Qdrant 配置
            'qdrant_host': 'localhost',
            'qdrant_port': '6333',
            'qdrant_collection_name': 'ros2_camera_images',
            
            # 通用配置
            'vector_dim': 512,
            'storage_interval': 5.0,
            'processing_frequency': 2.0,
            'web_viewer_port': 8889
        }
    
    def _create_gui(self):
        """創建 GUI 界面"""
        print("🔧 [GUI] Creating main frame...")
        # 主框架
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        print("🔧 [GUI] Creating title label...")
        # 標題
        title_label = ttk.Label(main_frame, text="🤖 ROS2 Image Processor GUI", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        print("🔧 [GUI] Creating notebook...")
        # 創建筆記本組件（分頁）
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        print("🔧 [GUI] Creating control page...")
        # 控制頁面
        self._create_control_page(notebook)
        
        print("🔧 [GUI] Creating status page...")
        # 狀態頁面
        self._create_status_page(notebook)
        
        print("🔧 [GUI] Creating config page...")
        # 配置頁面
        try:
            self._create_config_page(notebook)
            print("✅ [GUI] Config page created successfully")
        except Exception as e:
            print(f"❌ [GUI] Config page creation failed: {e}")
            # 創建一個簡化的配置頁面
            simple_config_frame = ttk.Frame(notebook)
            notebook.add(simple_config_frame, text="⚙️ 配置設定")
            ttk.Label(simple_config_frame, text="配置頁面暫時不可用").pack(pady=20)
        
        print("🔧 [GUI] Creating log page...")
        # 日誌頁面
        try:
            self._create_log_page(notebook)
            print("✅ [GUI] Log page created successfully")
        except Exception as e:
            print(f"❌ [GUI] Log page creation failed: {e}")
            # 創建簡化的日誌頁面
            simple_log_frame = ttk.Frame(notebook)
            notebook.add(simple_log_frame, text="📋 系統日誌")
            ttk.Label(simple_log_frame, text="日誌頁面暫時不可用").pack(pady=20)
        
        print("✅ [GUI] All GUI components created successfully")
    
    def _create_control_page(self, notebook):
        """創建控制頁面"""
        control_frame = ttk.Frame(notebook)
        notebook.add(control_frame, text="🎛️ 控制面板")
        
        # ROS2 連接控制
        ros2_group = ttk.LabelFrame(control_frame, text="ROS2 連接控制")
        ros2_group.pack(fill=tk.X, padx=10, pady=5)
        
        ros2_control_frame = ttk.Frame(ros2_group)
        ros2_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.ros2_start_btn = ttk.Button(ros2_control_frame, text="🔗 連接 ROS2", 
                                        command=self._start_ros2, width=15)
        self.ros2_start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.ros2_stop_btn = ttk.Button(ros2_control_frame, text="🔌 斷開 ROS2", 
                                       command=self._stop_ros2, width=15, state=tk.DISABLED)
        self.ros2_stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.ros2_status_label = ttk.Label(ros2_control_frame, text="狀態: 未連接", 
                                          foreground="red")
        self.ros2_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # 存儲控制
        storage_group = ttk.LabelFrame(control_frame, text="資料存儲控制")
        storage_group.pack(fill=tk.X, padx=10, pady=5)
        
        storage_control_frame = ttk.Frame(storage_group)
        storage_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.storage_start_btn = ttk.Button(storage_control_frame, text="💾 開始存儲", 
                                           command=self._start_storage, width=15, state=tk.DISABLED)
        self.storage_start_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.storage_stop_btn = ttk.Button(storage_control_frame, text="⏹️ 停止存儲", 
                                          command=self._stop_storage, width=15, state=tk.DISABLED)
        self.storage_stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.storage_status_label = ttk.Label(storage_control_frame, text="狀態: 未啟動", 
                                             foreground="orange")
        self.storage_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # 資料庫選擇
        db_selection_frame = ttk.Frame(storage_group)
        db_selection_frame.pack(fill=tk.X, padx=10, pady=(5, 0))
        
        ttk.Label(db_selection_frame, text="目標資料庫:", width=12).pack(side=tk.LEFT)
        
        self.db_type_var = tk.StringVar(value=self.config['database_type'])
        db_combo = ttk.Combobox(db_selection_frame, textvariable=self.db_type_var, 
                               values=['milvus', 'qdrant'], width=15, state="readonly")
        db_combo.pack(side=tk.LEFT, padx=(5, 10))
        
        # 資料庫狀態顯示
        self.current_db_status = ttk.Label(db_selection_frame, text="Milvus: 未連接", 
                                          font=('Arial', 9), foreground="gray")
        self.current_db_status.pack(side=tk.LEFT, padx=(10, 0))
        
        # 綁定選擇事件
        db_combo.bind('<<ComboboxSelected>>', self._on_database_selection_changed)
        
        # 手動存儲
        manual_storage_frame = ttk.Frame(storage_group)
        manual_storage_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.manual_store_btn = ttk.Button(manual_storage_frame, text="📷 手動存儲當前幀", 
                                          command=self._manual_store, width=20, state=tk.DISABLED)
        self.manual_store_btn.pack(side=tk.LEFT)
        
        # 資料庫連接控制（動態顯示）
        self.db_control_group = ttk.LabelFrame(control_frame, text=f"{self.current_db_type.upper()} 資料庫控制")
        self.db_control_group.pack(fill=tk.X, padx=10, pady=5)
        
        db_control_frame = ttk.Frame(self.db_control_group)
        db_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.db_connect_btn = ttk.Button(db_control_frame, text="🗃️ 連接資料庫", 
                                        command=self._connect_current_database, width=15)
        self.db_connect_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.db_disconnect_btn = ttk.Button(db_control_frame, text="🔌 斷開資料庫", 
                                           command=self._disconnect_current_database, width=15, state=tk.DISABLED)
        self.db_disconnect_btn.pack(side=tk.LEFT, padx=5)
        
        self.db_status_display = ttk.Label(db_control_frame, text="狀態: 未連接", 
                                          foreground="red")
        self.db_status_display.pack(side=tk.LEFT, padx=(20, 0))
        
        # 資料庫詳細信息
        db_info_frame = ttk.Frame(self.db_control_group)
        db_info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.db_info_label = ttk.Label(db_info_frame, text="Host: -- | Collection: -- | Records: --", 
                                      font=('Arial', 9), foreground="gray")
        self.db_info_label.pack(side=tk.LEFT)
        
        ttk.Button(db_info_frame, text="🔄 重新連接", 
                  command=self._reconnect_current_database, width=12).pack(side=tk.RIGHT)

        # Ollama 連接控制  
        ollama_group = ttk.LabelFrame(control_frame, text="Ollama AI 服務控制")
        ollama_group.pack(fill=tk.X, padx=10, pady=5)
        
        ollama_control_frame = ttk.Frame(ollama_group)
        ollama_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.ollama_check_btn = ttk.Button(ollama_control_frame, text="🤖 檢查 Ollama", 
                                          command=self._check_ollama, width=15)
        self.ollama_check_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.ollama_status_display = ttk.Label(ollama_control_frame, text="狀態: 未檢查", 
                                              foreground="orange")
        self.ollama_status_display.pack(side=tk.LEFT, padx=(20, 0))
        
        # Ollama 詳細信息
        ollama_info_frame = ttk.Frame(ollama_group)
        ollama_info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.ollama_info_label = ttk.Label(ollama_info_frame, text="URL: http://localhost:11434 | Model: --", 
                                          font=('Arial', 9), foreground="gray")
        self.ollama_info_label.pack(side=tk.LEFT)
        
        # 快速操作區
        quick_actions_group = ttk.LabelFrame(control_frame, text="快速操作")
        quick_actions_group.pack(fill=tk.X, padx=10, pady=5)
        
        quick_actions_frame = ttk.Frame(quick_actions_group)
        quick_actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(quick_actions_frame, text="🔄 檢查所有服務", 
                  command=self._check_all_services, width=15).pack(side=tk.LEFT, padx=(0, 5))
        
        self.web_viewer_btn = ttk.Button(quick_actions_frame, text="🌐 開啟 Web 查看器", 
                  command=self._open_web_viewer, width=15)
        self.web_viewer_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(quick_actions_frame, text="📊 刷新統計", 
                  command=self._refresh_stats, width=15).pack(side=tk.LEFT, padx=5)
    
    def _create_status_page(self, notebook):
        """創建狀態頁面"""
        status_frame = ttk.Frame(notebook)
        notebook.add(status_frame, text="📊 狀態監控")
        
        # 統計信息
        stats_group = ttk.LabelFrame(status_frame, text="統計信息")
        stats_group.pack(fill=tk.X, padx=10, pady=5)
        
        stats_frame = ttk.Frame(stats_group)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 左側統計
        left_stats = ttk.Frame(stats_frame)
        left_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.image_count_label = ttk.Label(left_stats, text="📷 接收影像: 0", font=('Arial', 12))
        self.image_count_label.pack(anchor=tk.W, pady=2)
        
        self.tf_count_label = ttk.Label(left_stats, text="📡 接收 TF: 0", font=('Arial', 12))
        self.tf_count_label.pack(anchor=tk.W, pady=2)
        
        self.stored_count_label = ttk.Label(left_stats, text="💾 已存儲: 0", font=('Arial', 12))
        self.stored_count_label.pack(anchor=tk.W, pady=2)
        
        # 右側統計
        right_stats = ttk.Frame(stats_frame)
        right_stats.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.last_image_label = ttk.Label(right_stats, text="📷 最後影像: --", font=('Arial', 10))
        self.last_image_label.pack(anchor=tk.W, pady=2)
        
        self.last_tf_label = ttk.Label(right_stats, text="📡 最後 TF: --", font=('Arial', 10))
        self.last_tf_label.pack(anchor=tk.W, pady=2)
        
        self.last_storage_label = ttk.Label(right_stats, text="💾 最後存儲: --", font=('Arial', 10))
        self.last_storage_label.pack(anchor=tk.W, pady=2)
        
        # 當前數據包信息
        current_data_group = ttk.LabelFrame(status_frame, text="當前數據包信息")
        current_data_group.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 數據包信息文本區域
        self.data_info_text = scrolledtext.ScrolledText(current_data_group, height=8, width=80)
        self.data_info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 實時影像預覽（如果有 PIL）
        if PIL_AVAILABLE:
            preview_group = ttk.LabelFrame(status_frame, text="影像預覽")
            preview_group.pack(fill=tk.X, padx=10, pady=5)
            
            self.image_preview_label = ttk.Label(preview_group, text="無影像", anchor=tk.CENTER)
            self.image_preview_label.pack(pady=10)
    
    def _create_config_page(self, notebook):
        """創建配置頁面"""
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="⚙️ 配置設定")
        
        # 配置滾動框
        canvas = tk.Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # ROS2 配置
        ros2_config_group = ttk.LabelFrame(scrollable_frame, text="ROS2 配置")
        ros2_config_group.pack(fill=tk.X, padx=10, pady=5)
        
        # 影像話題
        image_topic_frame = ttk.Frame(ros2_config_group)
        image_topic_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(image_topic_frame, text="影像話題:", width=15).pack(side=tk.LEFT)
        self.image_topic_var = tk.StringVar(value=self.config['image_topic'])
        ttk.Entry(image_topic_frame, textvariable=self.image_topic_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        # TF 話題
        tf_topic_frame = ttk.Frame(ros2_config_group)
        tf_topic_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(tf_topic_frame, text="TF 話題:", width=15).pack(side=tk.LEFT)
        self.tf_topic_var = tk.StringVar(value=self.config['tf_topic'])
        ttk.Entry(tf_topic_frame, textvariable=self.tf_topic_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        # 目標框架
        target_frame_frame = ttk.Frame(ros2_config_group)
        target_frame_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(target_frame_frame, text="目標框架:", width=15).pack(side=tk.LEFT)
        self.target_frame_var = tk.StringVar(value=self.config['target_frame'])
        ttk.Entry(target_frame_frame, textvariable=self.target_frame_var, width=40).pack(side=tk.LEFT, padx=(5, 0))
        
        # Milvus 配置
        milvus_config_group = ttk.LabelFrame(scrollable_frame, text="Milvus 配置")
        milvus_config_group.pack(fill=tk.X, padx=10, pady=5)
        
        # Milvus 主機
        milvus_host_frame = ttk.Frame(milvus_config_group)
        milvus_host_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(milvus_host_frame, text="主機:", width=15).pack(side=tk.LEFT)
        self.milvus_host_var = tk.StringVar(value=self.config['milvus_host'])
        ttk.Entry(milvus_host_frame, textvariable=self.milvus_host_var, width=20).pack(side=tk.LEFT, padx=(5, 0))
        
        # Milvus 端口
        milvus_port_frame = ttk.Frame(milvus_config_group)
        milvus_port_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(milvus_port_frame, text="端口:", width=15).pack(side=tk.LEFT)
        self.milvus_port_var = tk.StringVar(value=self.config['milvus_port'])
        ttk.Entry(milvus_port_frame, textvariable=self.milvus_port_var, width=20).pack(side=tk.LEFT, padx=(5, 0))
        
        # 集合名稱
        collection_name_frame = ttk.Frame(milvus_config_group)
        collection_name_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(collection_name_frame, text="集合名稱:", width=15).pack(side=tk.LEFT)
        self.collection_name_var = tk.StringVar(value=self.config['collection_name'])
        ttk.Entry(collection_name_frame, textvariable=self.collection_name_var, width=30).pack(side=tk.LEFT, padx=(5, 0))
        
        # Ollama 配置
        ollama_config_group = ttk.LabelFrame(scrollable_frame, text="Ollama 配置")
        ollama_config_group.pack(fill=tk.X, padx=10, pady=5)
        
        # Ollama URL
        ollama_url_frame = ttk.Frame(ollama_config_group)
        ollama_url_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ollama_url_frame, text="Ollama URL:", width=15).pack(side=tk.LEFT)
        self.ollama_url_var = tk.StringVar(value=self.config['ollama_url'])
        ttk.Entry(ollama_url_frame, textvariable=self.ollama_url_var, width=30).pack(side=tk.LEFT, padx=(5, 0))
        
        # Ollama 模型
        ollama_model_frame = ttk.Frame(ollama_config_group)
        ollama_model_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ollama_model_frame, text="模型:", width=15).pack(side=tk.LEFT)
        self.ollama_model_var = tk.StringVar(value=self.config['ollama_model'])
        ttk.Entry(ollama_model_frame, textvariable=self.ollama_model_var, width=30).pack(side=tk.LEFT, padx=(5, 0))
        
        # AI 分析開關
        ai_enable_frame = ttk.Frame(ollama_config_group)
        ai_enable_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ai_enable_frame, text="啟用 AI 分析:", width=15).pack(side=tk.LEFT)
        self.ai_enable_var = tk.BooleanVar(value=self.config['enable_ai_analysis'])
        ttk.Checkbutton(ai_enable_frame, variable=self.ai_enable_var).pack(side=tk.LEFT, padx=(5, 0))
        
        # 配置按鈕
        config_buttons_frame = ttk.Frame(scrollable_frame)
        config_buttons_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(config_buttons_frame, text="💾 保存配置", 
                  command=self._save_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(config_buttons_frame, text="📁 載入配置", 
                  command=self._load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons_frame, text="🔄 應用配置", 
                  command=self._apply_config).pack(side=tk.LEFT, padx=5)
        
        # 打包配置頁面
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_log_page(self, notebook):
        """創建簡化的系統信息頁面（主要日誌在 terminal）"""
        print("🔧 [LOG] Creating log frame...")
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="📝 系統信息")
        
        print("🔧 [LOG] Creating info label...")
        # 說明文字
        info_label = ttk.Label(log_frame, 
                              text="💡 系統日誌主要顯示在 Terminal 中\n請查看啟動應用程式的終端視窗", 
                              font=('Arial', 12), 
                              foreground="blue")
        info_label.pack(pady=20)
        
        print("✅ [LOG] Basic log page created successfully")
        
        # 簡化的日誌顯示（最近的幾條）
        ttk.Label(log_frame, text="最近的系統訊息:", font=('Arial', 10, 'bold')).pack(pady=(20, 5))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        
        # 日誌控制
        log_control_frame = ttk.Frame(log_frame)
        log_control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(log_control_frame, text="🗑️ 清除", 
                  command=self._clear_logs).pack(side=tk.LEFT)
        ttk.Button(log_control_frame, text="💾 保存", 
                  command=self._save_logs).pack(side=tk.LEFT, padx=(5, 0))
        
        # 添加初始訊息
        self._add_log("🚀 ROS2 Image Processor GUI 已啟動 - 主要日誌顯示在 Terminal")
    
    def _on_database_selection_changed(self, event=None):
        """資料庫選擇變更處理"""
        new_db_type = self.db_type_var.get()
        
        if new_db_type != self.current_db_type:
            self._add_log(f"🔄 切換目標資料庫從 {self.current_db_type} 到 {new_db_type}")
            self.current_db_type = new_db_type
            self.config['database_type'] = new_db_type
            
            # 更新資料庫控制區域的UI顯示
            self._update_db_control_ui()
            
            # 更新當前資料庫狀態顯示
            self._update_current_db_status()
            
            # 如果新資料庫尚未連接，提示用戶
            if new_db_type not in self.db_managers or not self.db_managers[new_db_type].connected:
                self._add_log(f"💡 請先連接 {new_db_type.upper()} 資料庫以啟用存儲功能")
                self._add_log(f"💡 請先連接 {new_db_type.upper()} 資料庫以啟用存儲功能")
    
    def _get_current_db_manager(self) -> VectorDBManager:
        """獲取當前選擇的資料庫管理器"""
        db_type = self.current_db_type
        
        if db_type not in self.db_managers:
            # 創建對應的資料庫管理器
            self.db_managers[db_type] = DatabaseManagerFactory.create_manager(
                db_type, self.config, self.status_queue
            )
        
        return self.db_managers[db_type]
    
    def _update_current_db_status(self):
        """更新當前資料庫狀態顯示"""
        try:
            if self.current_db_type in self.db_managers:
                db_manager = self.db_managers[self.current_db_type]
                status_text = "已連接" if db_manager.connected else "未連接"
                color = "green" if db_manager.connected else "red"
            else:
                status_text = "未連接"
                color = "red"
            
            self.current_db_status.config(
                text=f"{self.current_db_type.upper()}: {status_text}",
                foreground=color
            )
        except Exception as e:
            self._add_log(f"❌ 更新資料庫狀態錯誤: {e}")
    
    def _update_db_control_ui(self):
        """更新資料庫控制區域的UI顯示"""
        try:
            # 更新控制區域標題
            self.db_control_group.config(text=f"{self.current_db_type.upper()} 資料庫控制")
            
            # 更新資料庫信息顯示
            self._update_db_info_display()
            
            # 更新按鈕狀態
            if self.current_db_type in self.db_managers and self.db_managers[self.current_db_type].connected:
                self.db_connect_btn.config(state=tk.DISABLED)
                self.db_disconnect_btn.config(state=tk.NORMAL)
                self.db_status_display.config(text="狀態: 已連接", foreground="green")
            else:
                self.db_connect_btn.config(state=tk.NORMAL)
                self.db_disconnect_btn.config(state=tk.DISABLED)
                self.db_status_display.config(text="狀態: 未連接", foreground="red")
                
        except Exception as e:
            self._add_log(f"❌ 更新資料庫控制UI錯誤: {e}")
    
    def _update_db_info_display(self):
        """更新資料庫信息顯示"""
        try:
            if self.current_db_type in self.db_managers:
                db_manager = self.db_managers[self.current_db_type]
                if db_manager.connected:
                    if self.current_db_type == 'milvus':
                        info_text = f"Host: {db_manager.host}:{db_manager.port} | Collection: {db_manager.collection_name} | Records: {db_manager.stored_count}"
                    elif self.current_db_type == 'qdrant':
                        info_text = f"Host: {db_manager.host}:{db_manager.port} | Collection: {db_manager.collection_name} | Points: {db_manager.stored_count}"
                    else:
                        info_text = f"Host: {db_manager.host}:{db_manager.port} | Connected"
                else:
                    if self.current_db_type == 'milvus':
                        info_text = f"Host: {self.config.get('milvus_host', 'localhost')}:{self.config.get('milvus_port', '19530')} | Collection: -- | Records: --"
                    elif self.current_db_type == 'qdrant':
                        info_text = f"Host: {self.config.get('qdrant_host', 'localhost')}:{self.config.get('qdrant_port', '6333')} | Collection: -- | Points: --"
                    else:
                        info_text = "Host: -- | Collection: -- | Records: --"
            else:
                if self.current_db_type == 'milvus':
                    info_text = f"Host: {self.config.get('milvus_host', 'localhost')}:{self.config.get('milvus_port', '19530')} | Collection: -- | Records: --"
                elif self.current_db_type == 'qdrant':
                    info_text = f"Host: {self.config.get('qdrant_host', 'localhost')}:{self.config.get('qdrant_port', '6333')} | Collection: -- | Points: --"
                else:
                    info_text = "Host: -- | Collection: -- | Records: --"
            
            self.db_info_label.config(text=info_text)
            
        except Exception as e:
            self._add_log(f"❌ 更新資料庫信息顯示錯誤: {e}")
    
    def _connect_current_database(self):
        """連接當前選擇的資料庫"""
        self._connect_database(self.current_db_type)
    
    def _disconnect_current_database(self):
        """斷開當前選擇的資料庫"""
        self._disconnect_database(self.current_db_type)
    
    def _reconnect_current_database(self):
        """重新連接當前選擇的資料庫"""
        self._reconnect_database(self.current_db_type)
    
    def _add_log(self, message):
        """添加日誌到 terminal"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        # 同時更新 GUI 日誌（如果日誌頁面存在）
        if hasattr(self, 'log_text'):
            try:
                self.log_text.insert(tk.END, log_entry + "\n")
                self.log_text.see(tk.END)
            except:
                pass  # 忽略 GUI 日誌錯誤
    
    def _clear_logs(self):
        """清除日誌"""
        self.log_text.delete(1.0, tk.END)
        self._add_log("📝 日誌已清除")
    
    def _save_logs(self):
        """保存日誌"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self._add_log(f"💾 日誌已保存到: {filename}")
            except Exception as e:
                self._add_log(f"❌ 保存日誌失敗: {e}")
    
    def _open_web_viewer(self):
        """開啟 Web 查看器"""
        try:
            # 启动内置Web服务器
            if not self.web_server_running:
                self._start_web_server()
                # 等待服务器启动
                time.sleep(1)
            
            # 更新Web数据
            self._export_images_for_web()
            
            # 打开浏览器
            url = f"http://localhost:{self.web_server_port}"
            webbrowser.open(url)
            self._add_log(f"🌐 Web 查看器已開啟: {url}")
            
            # 更新按钮状态
            if hasattr(self, 'web_viewer_btn'):
                self.web_viewer_btn.config(text=f"🌐 查看器 (:{self.web_server_port})")
            
        except Exception as e:
            self._add_log(f"❌ 開啟 Web 查看器失敗: {e}")
            messagebox.showerror("错误", f"启动Web查看器失败:\n{e}")
    
    def _check_stored_images(self):
        """检查是否有存储的图片"""
        try:
            current_db_manager = self._get_current_db_manager()
            if current_db_manager.connected:
                count = current_db_manager.get_total_count()
                return count > 0
            return False
        except:
            return False
    
    def _start_web_server(self):
        """启动内置Web服务器"""
        try:
            # 创建Web查看器文件
            self._create_web_viewer_files()
            
            # 启动简单的HTTP服务器线程
            self.web_server_thread = threading.Thread(target=self._run_simple_web_server, daemon=True)
            self.web_server_thread.start()
            
            self.web_server_running = True
            self._add_log(f"🌐 Web服务器已在端口 {self.web_server_port} 启动")
            
        except Exception as e:
            self._add_log(f"❌ 启动Web服务器失败: {e}")
    
    def _run_simple_web_server(self):
        """运行简单的Web服务器"""
        import http.server
        import socketserver
        import os
        
        # 切换到查看器目录
        viewer_dir = "/tmp/ros2_image_viewer"
        os.chdir(viewer_dir)
        
        # 创建自定义的请求处理器
        class CustomHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/api/images':
                    # 提供API数据
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    try:
                        with open('api_images.json', 'r', encoding='utf-8') as f:
                            self.wfile.write(f.read().encode('utf-8'))
                    except:
                        self.wfile.write(b'{"total": 0, "images": []}')
                else:
                    # 默认处理静态文件
                    super().do_GET()
        
        try:
            with socketserver.TCPServer(("", self.web_server_port), CustomHandler) as httpd:
                httpd.serve_forever()
        except Exception as e:
            if self.web_server_running:
                self.status_queue.put(('log', f"❌ Web服务器错误: {e}"))
    
    def _create_web_viewer_files(self):
        """创建Web查看器所需的文件"""
        import os
        
        # 创建临时目录
        viewer_dir = "/tmp/ros2_image_viewer"
        os.makedirs(viewer_dir, exist_ok=True)
        
        # 生成HTML页面
        html_content = self._generate_html_content()
        
        with open(os.path.join(viewer_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # 导出图片数据
        self._export_images_for_web()
    
    def _generate_html_content(self):
        """生成HTML查看器内容"""
        return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ROS2 图片查看器</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #2196F3;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .image-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .image-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .image-info {
            font-size: 12px;
            color: #666;
            line-height: 1.4;
        }
        .ai-analysis {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 11px;
        }
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 18px;
            color: #666;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #2196F3;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .refresh-btn:hover {
            background: #1976D2;
        }
    </style>
</head>
<body>
    <button class="refresh-btn" onclick="location.reload()">🔄 刷新</button>
    
    <div class="header">
        <h1>🤖 ROS2 图片查看器</h1>
        <p>查看存储在数据库中的图片和AI分析结果</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number" id="total-count">-</div>
            <div>总图片数</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="db-type">-</div>
            <div>数据库类型</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="last-update">-</div>
            <div>最后更新</div>
        </div>
    </div>
    
    <div id="gallery" class="gallery">
        <div class="loading">
            📷 正在加载图片数据...
        </div>
    </div>

    <script>
        async function loadImages() {
            try {
                const response = await fetch('/api/images');
                const data = await response.json();
                
                // 更新统计信息
                document.getElementById('total-count').textContent = data.total || 0;
                document.getElementById('db-type').textContent = data.db_type || '-';
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                
                // 显示图片
                const gallery = document.getElementById('gallery');
                if (data.images && data.images.length > 0) {
                    gallery.innerHTML = data.images.map(img => `
                        <div class="image-card">
                            <img src="data:image/jpeg;base64,${img.image_base64}" alt="ROS2 Image" />
                            <div class="image-info">
                                <strong>📅 时间:</strong> ${img.timestamp}<br>
                                <strong>📍 位置:</strong> [${img.position.map(p => p.toFixed(3)).join(', ')}]<br>
                                <strong>🔄 旋转:</strong> [${img.rotation.map(r => r.toFixed(3)).join(', ')}]<br>
                                <strong>📐 尺寸:</strong> ${img.image_size}<br>
                                <strong>🆔 ID:</strong> ${img.frame_id || 'N/A'}
                            </div>
                            ${img.ai_analysis_success && img.ai_answer ? `
                                <div class="ai-analysis">
                                    <strong>🤖 AI分析:</strong><br>
                                    ${img.ai_answer}
                                </div>
                            ` : ''}
                        </div>
                    `).join('');
                } else {
                    gallery.innerHTML = '<div class="loading">📷 暂无图片数据</div>';
                }
            } catch (error) {
                console.error('加载图片失败:', error);
                document.getElementById('gallery').innerHTML = 
                    '<div class="loading">❌ 加载图片失败，请检查服务器状态</div>';
            }
        }
        
        // 页面加载时获取图片
        loadImages();
        
        // 每30秒自动刷新
        setInterval(loadImages, 30000);
    </script>
</body>
</html>
        """
    
    def _export_images_for_web(self):
        """导出图片数据供Web查看器使用"""
        try:
            import json
            import os
            
            current_db_manager = self._get_current_db_manager()
            if not current_db_manager.connected:
                # 创建空数据
                api_data = {
                    "total": 0,
                    "db_type": self.current_db_type.upper(),
                    "timestamp": datetime.now().isoformat(),
                    "images": []
                }
            else:
                # 获取图片数据
                images_data = self._get_images_for_web_viewer()
                
                # 创建API响应
                api_data = {
                    "total": len(images_data),
                    "db_type": self.current_db_type.upper(),
                    "timestamp": datetime.now().isoformat(),
                    "images": images_data
                }
            
            # 保存到文件供Web服务器使用
            viewer_dir = "/tmp/ros2_image_viewer"
            with open(os.path.join(viewer_dir, "api_images.json"), "w", encoding="utf-8") as f:
                json.dump(api_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self._add_log(f"❌ 导出图片数据失败: {e}")
    
    def _get_images_for_web_viewer(self):
        """获取图片数据用于Web查看器"""
        try:
            current_db_manager = self._get_current_db_manager()
            
            if self.current_db_type == 'milvus':
                return self._get_milvus_images_for_web()
            elif self.current_db_type == 'qdrant':
                return self._get_qdrant_images_for_web()
            else:
                return []
                
        except Exception as e:
            self._add_log(f"❌ 获取图片数据失败: {e}")
            return []
    
    def _get_milvus_images_for_web(self):
        """从Milvus获取图片数据"""
        try:
            import json
            
            current_db_manager = self._get_current_db_manager()
            if not current_db_manager.connected:
                return []
            
            # 查询最近的20张图片
            collection = current_db_manager.collection
            
            # 先获取总数量
            total_count = collection.num_entities
            if total_count == 0:
                return []
            
            # 获取最近的数据
            limit = min(20, total_count)
            results = collection.query(
                expr="",
                output_fields=["timestamp", "image_base64", "ai_answer", "ai_analysis_success", 
                              "position_x", "position_y", "position_z", 
                              "rotation_x", "rotation_y", "rotation_z", "rotation_w",
                              "image_size", "metadata"],
                limit=limit
            )
            
            images = []
            for result in results:
                try:
                    metadata_str = result.get('metadata', '{}')
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    
                    images.append({
                        "timestamp": result.get('timestamp', ''),
                        "image_base64": result.get('image_base64', ''),
                        "ai_answer": result.get('ai_answer', ''),
                        "ai_analysis_success": result.get('ai_analysis_success', False),
                        "position": [
                            result.get('position_x', 0.0),
                            result.get('position_y', 0.0),
                            result.get('position_z', 0.0)
                        ],
                        "rotation": [
                            result.get('rotation_w', 1.0),
                            result.get('rotation_x', 0.0),
                            result.get('rotation_y', 0.0),
                            result.get('rotation_z', 0.0)
                        ],
                        "image_size": result.get('image_size', ''),
                        "frame_id": metadata.get('frame_id', 'N/A')
                    })
                except Exception as e:
                    continue
            
            return images
            
        except Exception as e:
            self._add_log(f"❌ 从Milvus获取图片失败: {e}")
            return []
    
    def _get_qdrant_images_for_web(self):
        """从Qdrant获取图片数据"""
        try:
            current_db_manager = self._get_current_db_manager()
            if not current_db_manager.connected:
                return []
            
            # 获取最近的20个点
            results = current_db_manager.client.scroll(
                collection_name=current_db_manager.collection_name,
                limit=20,
                with_payload=True,
                with_vectors=False
            )
            
            images = []
            for point in results[0]:  # results is (points, next_page_offset)
                try:
                    payload = point.payload
                    images.append({
                        "timestamp": payload.get('timestamp', ''),
                        "image_base64": payload.get('image_base64', ''),
                        "ai_answer": payload.get('ai_answer', ''),
                        "ai_analysis_success": payload.get('ai_analysis_success', False),
                        "position": [
                            payload.get('position_x', 0.0),
                            payload.get('position_y', 0.0),
                            payload.get('position_z', 0.0)
                        ],
                        "rotation": [
                            payload.get('rotation_w', 1.0),
                            payload.get('rotation_x', 0.0),
                            payload.get('rotation_y', 0.0),
                            payload.get('rotation_z', 0.0)
                        ],
                        "image_size": payload.get('image_size', ''),
                        "frame_id": payload.get('frame_id', 'N/A')
                    })
                except:
                    continue
            
            return images
            
        except Exception as e:
            self._add_log(f"❌ 从Qdrant获取图片失败: {e}")
            return []
    
    def _manual_store(self):
        """手動存儲當前幀（增加確認）"""
        
        # 檢查當前資料庫連接狀態
        db_manager = self._get_current_db_manager()
        if not db_manager.connected:
            response = messagebox.askyesno(
                "資料庫未連接", 
                f"當前選擇的 {self.current_db_type.upper()} 資料庫未連接。\n是否要先連接該資料庫？"
            )
            if response:
                # 嘗試連接
                if self.current_db_type == 'milvus':
                    self._connect_database('milvus')
                elif self.current_db_type == 'qdrant':
                    self._connect_database('qdrant')
            return
        
        # 確認存儲
        response = messagebox.askyesno(
            "確認手動存儲", 
            f"即將存儲當前影像到 {self.current_db_type.upper()} 資料庫\n是否繼續？"
        )
        
        if response:
            if self._store_current_data():
                self._add_log(f"📷 手動存儲到 {self.current_db_type.upper()} 完成")
            else:
                self._add_log(f"❌ 手動存儲到 {self.current_db_type.upper()} 失敗")
    
    def _connect_database(self, db_type: str):
        """通用資料庫連接方法"""
        try:
            self._add_log(f"🗃️ 正在連接 {db_type.upper()}...")
            
            # 應用當前配置
            self._apply_config()
            
            # 獲取或創建資料庫管理器
            db_manager = self._get_current_db_manager()
            
            # 嘗試連接
            success = db_manager.connect()
            
            if success:
                self._add_log(f"✅ {db_type.upper()} 連接成功")
                self._update_db_control_ui()  # 更新UI顯示
                self._update_current_db_status()
                
                # 如果 ROS2 已連接，啟用存儲功能
                if self.ros2_running:
                    self.storage_start_btn.config(state=tk.NORMAL)
                    self.manual_store_btn.config(state=tk.NORMAL)
            else:
                self._add_log(f"❌ {db_type.upper()} 連接失敗")
                
        except Exception as e:
            self._add_log(f"❌ {db_type.upper()} 連接錯誤: {e}")
    
    def _store_current_data(self):
        """存儲當前數據包（修改版）"""
        try:
            with self.data_lock:
                if self.current_data_packet is None:
                    return False
                data_packet = self.current_data_packet
            
            # 獲取當前選擇的資料庫管理器
            db_manager = self._get_current_db_manager()
            
            if not db_manager.connected:
                self._add_log(f"❌ {self.current_db_type.upper()} 資料庫未連接")
                return False
            
            # VLM 分析
            ai_answer = "AI analysis disabled"
            if self.config['enable_ai_analysis'] and self.vlm_analyzer and self.status_manager.ollama_connected:
                ai_answer = self.vlm_analyzer.analyze_image(data_packet.image)
            
            # 存儲到選擇的資料庫
            self._add_log(f"💾 正在存儲到 {self.current_db_type.upper()} 資料庫...")
            success = db_manager.store_data_packet(data_packet, ai_answer)
            
            if success:
                self._add_log(f"✅ 成功存儲到 {self.current_db_type.upper()}")
            
            return success
            
        except Exception as e:
            self.status_queue.put(('error', f"Store data error: {e}"))
            return False
    
    def _start_status_update_loop(self):
        """啟動狀態更新循環"""
        self._update_status()
        self.master.after(100, self._start_status_update_loop)
    
    def _update_status(self):
        """更新狀態顯示"""
        try:
            while not self.status_queue.empty():
                msg_type, data = self.status_queue.get_nowait()
                
                if msg_type == 'log':
                    self._add_log(data)
                elif msg_type == 'image_received':
                    self._update_image_status(data)
                elif msg_type == 'tf_received':
                    self._update_tf_status(data)
                elif msg_type == 'data_stored':
                    self._update_storage_status(data)
                elif msg_type == 'milvus_connected':
                    self._update_database_status('milvus', data)
                elif msg_type == 'qdrant_connected':
                    self._update_database_status('qdrant', data)
                elif msg_type == 'ollama_connected':
                    self._update_ollama_status(data)
                elif msg_type == 'error':
                    self._add_log(f"❌ {data}")
                    
        except queue.Empty:
            pass
    
    def _update_image_status(self, data):
        """更新影像狀態"""
        self.status_manager.image_count = data['count']
        self.status_manager.last_image_time = data['timestamp']
        self.status_manager.image_received = True
        
        self.image_count_label.config(text=f"📷 接收影像: {data['count']}")
        self.last_image_label.config(text=f"📷 最後影像: {data['timestamp'].strftime('%H:%M:%S')}")
        
        # 更新預覽（如果有 PIL）
        if PIL_AVAILABLE and hasattr(self, 'image_preview_label'):
            self._update_image_preview()
    
    def _update_tf_status(self, data):
        """更新 TF 狀態"""
        self.status_manager.tf_count = data['count']
        self.status_manager.last_tf_time = data['timestamp']
        self.status_manager.tf_received = True
        
        self.tf_count_label.config(text=f"📡 接收 TF: {data['count']}")
        self.last_tf_label.config(text=f"📡 最後 TF: {data['timestamp'].strftime('%H:%M:%S')}")
        
        # 更新當前數據包信息
        self._update_current_data_info(data)
    
    def _update_storage_status(self, data):
        """更新存儲狀態"""
        self.status_manager.stored_count = data['count']
        self.status_manager.last_storage_time = data['timestamp']
        
        self.stored_count_label.config(text=f"💾 已存儲: {data['count']}")
        self.last_storage_label.config(text=f"💾 最後存儲: {data['timestamp'].strftime('%H:%M:%S')}")
    
    def _update_database_status(self, db_type: str, connected: bool):
        """更新資料庫狀態"""
        if db_type == 'milvus':
            self.status_manager.milvus_connected = connected
        elif db_type == 'qdrant':
            self.status_manager.qdrant_connected = connected
        
        # 更新當前選擇的資料庫狀態顯示
        if db_type == self.current_db_type:
            self._update_db_control_ui()  # 更新UI顯示
            self._update_current_db_status()
    
    def _connect_milvus(self):
        """手動連接 Milvus"""
        self._connect_database('milvus')
    
    def _disconnect_milvus(self):
        """手動斷開 Milvus"""
        self._disconnect_database('milvus')
    
    def _connect_qdrant(self):
        """手動連接 Qdrant"""
        self._connect_database('qdrant')
    
    def _disconnect_qdrant(self):
        """手動斷開 Qdrant"""
        self._disconnect_database('qdrant')
    
    def _disconnect_database(self, db_type: str):
        """通用資料庫斷開方法"""
        try:
            if db_type in self.db_managers and self.db_managers[db_type].connected:
                # 先停止存儲
                if self.storage_running:
                    self._stop_storage()
                
                # 斷開連接
                success = self.db_managers[db_type].disconnect()
                
                if success:
                    self._add_log(f"🔌 {db_type.upper()} 已手動斷開")
                    self._update_db_control_ui()  # 更新UI顯示
                    self._update_current_db_status()
                    
                    # 禁用存儲功能
                    self.storage_start_btn.config(state=tk.DISABLED)
                    self.storage_stop_btn.config(state=tk.DISABLED)
                else:
                    self._add_log(f"❌ {db_type.upper()} 斷開失敗")
            else:
                self._add_log(f"⚠️ {db_type.upper()} 未連接")
                
        except Exception as e:
            self._add_log(f"❌ {db_type.upper()} 斷開錯誤: {e}")
    
    def _reconnect_milvus(self):
        """重新連接 Milvus"""
        self._reconnect_database('milvus')
    
    def _reconnect_database(self, db_type: str):
        """通用資料庫重新連接方法"""
        try:
            self._add_log(f"🔄 正在重新連接 {db_type.upper()}...")
            
            # 先斷開
            if db_type in self.db_managers and self.db_managers[db_type].connected:
                self.db_managers[db_type].disconnect()
            
            # 等待一下
            time.sleep(1)
            
            # 重新連接
            self._connect_database(db_type)
            
        except Exception as e:
            self._add_log(f"❌ {db_type.upper()} 重新連接錯誤: {e}")
    
    def _check_ollama(self):
        """檢查 Ollama 服務"""
        try:
            self._add_log("🤖 正在檢查 Ollama 服務...")
            
            # 應用當前配置
            self._apply_config()
            
            # 如果還沒有 VLM analyzer，創建一個
            if not self.vlm_analyzer:
                self.vlm_analyzer = OllamaVLMAnalyzer(self.config, self.status_queue)
            else:
                # 重新檢查連接
                self.vlm_analyzer._check_connection()
            
            # 更新詳細信息
            self._update_ollama_info()
            
        except Exception as e:
            self._add_log(f"❌ Ollama 檢查錯誤: {e}")
    
    def _check_all_services(self):
        """檢查所有外部服務"""
        self._add_log("🔄 正在檢查所有外部服務...")
        
        # 檢查當前選擇的資料庫（如果未連接）
        current_db_manager = self._get_current_db_manager()
        if not current_db_manager.connected:
            threading.Thread(target=self._connect_database, args=(self.current_db_type,), daemon=True).start()
        
        # 檢查 Ollama
        threading.Thread(target=self._check_ollama, daemon=True).start()
    
    def _refresh_stats(self):
        """刷新統計信息"""
        try:
            self._add_log("📊 正在刷新統計信息...")
            
            # 刷新當前資料庫統計
            current_db_manager = self._get_current_db_manager()
            if current_db_manager.connected:
                old_count = current_db_manager.stored_count
                new_count = current_db_manager.get_total_count()
                if new_count != old_count:
                    current_db_manager.stored_count = new_count
                    self._add_log(f"📊 {self.current_db_type.upper()} 記錄數更新: {old_count} → {new_count}")
                
                self._update_db_control_ui()  # 更新UI顯示
                self._update_current_db_status()
            
            self._add_log("✅ 統計信息已刷新")
            
        except Exception as e:
            self._add_log(f"❌ 刷新統計錯誤: {e}")
    
    def _update_ollama_info(self):
        """更新 Ollama 詳細信息"""
        try:
            if self.vlm_analyzer:
                url = self.vlm_analyzer.ollama_url
                model = self.vlm_analyzer.ollama_model if self.vlm_analyzer.connected else "--"
                
                info_text = f"URL: {url} | Model: {model}"
                self.ollama_info_label.config(text=info_text)
            else:
                self.ollama_info_label.config(text="URL: -- | Model: --")
        except Exception as e:
            self._add_log(f"❌ 更新 Ollama 信息錯誤: {e}")
    
    def _update_ollama_status(self, connected):
        """更新 Ollama 狀態"""
        self.status_manager.ollama_connected = connected
        if connected:
            self.ollama_status_display.config(text="狀態: 已連接", foreground="green")
        else:
            self.ollama_status_display.config(text="狀態: 未連接", foreground="red")
        
        # 更新詳細信息
        self._update_ollama_info()
    
    def _update_current_data_info(self, tf_data):
        """更新當前數據包信息"""
        try:
            info_text = "=== 當前數據包信息 ===\n\n"
            
            # 影像信息
            if self.status_manager.image_received:
                info_text += f"📷 影像狀態: 已接收 {self.status_manager.image_count} 幀\n"
                info_text += f"📷 最後更新: {self.status_manager.last_image_time.strftime('%H:%M:%S') if self.status_manager.last_image_time else 'N/A'}\n\n"
            else:
                info_text += "📷 影像狀態: 未接收\n\n"
            
            # TF 信息
            if self.status_manager.tf_received:
                info_text += f"📡 TF 狀態: 已接收 {self.status_manager.tf_count} 條\n"
                info_text += f"📡 最後更新: {self.status_manager.last_tf_time.strftime('%H:%M:%S') if self.status_manager.last_tf_time else 'N/A'}\n"
                info_text += f"📡 找到目標: {tf_data.get('found_target', False)}\n"
                info_text += f"📡 可用框架: {', '.join(tf_data.get('available_frames', []))}\n\n"
            else:
                info_text += "📡 TF 狀態: 未接收\n\n"
            
            # 同步狀態
            if self.status_manager.image_received and self.status_manager.tf_received:
                info_text += "🔗 同步狀態: ✅ 可同步\n"
            else:
                info_text += "🔗 同步狀態: ❌ 等待數據\n"
            
            # 存儲狀態
            info_text += f"💾 存儲狀態: {'🟢 運行中' if self.storage_running else '🔴 未啟動'}\n"
            info_text += f"💾 已存儲: {self.status_manager.stored_count}\n"
            info_text += f"💾 目標資料庫: {self.current_db_type.upper()}\n"
            
            # 服務狀態
            info_text += f"\n=== 服務狀態 ===\n"
            info_text += f"🗃️ Milvus: {'✅ 已連接' if self.status_manager.milvus_connected else '❌ 未連接'}\n"
            info_text += f"🗄️ Qdrant: {'✅ 已連接' if self.status_manager.qdrant_connected else '❌ 未連接'}\n"
            info_text += f"🤖 Ollama: {'✅ 已連接' if self.status_manager.ollama_connected else '❌ 未連接'}\n"
            
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, info_text)
            
        except Exception as e:
            self._add_log(f"❌ 更新數據信息失敗: {e}")
    
    def _update_image_preview(self):
        """更新影像預覽"""
        try:
            if self.image_subscriber and PIL_AVAILABLE:
                image, timestamp = self.image_subscriber.get_latest_image()
                if image is not None:
                    # 縮放圖像以適應預覽
                    pil_image = Image.fromarray(image)
                    pil_image.thumbnail((200, 150), Image.Resampling.LANCZOS)
                    
                    # 轉換為 Tkinter 可用格式
                    photo = ImageTk.PhotoImage(pil_image)
                    self.image_preview_label.config(image=photo, text="")
                    self.image_preview_label.image = photo  # 保持引用
        except Exception as e:
            pass  # 忽略預覽錯誤
    
    def _start_ros2(self):
        """啟動 ROS2 連接"""
        if not ROS2_AVAILABLE:
            messagebox.showerror("錯誤", "ROS2 模組不可用，請檢查安裝")
            return
        
        if self.ros2_running:
            return
        
        try:
            self._add_log("🔗 正在連接 ROS2...")
            
            # 應用當前配置
            self._apply_config()
            
            # 初始化 ROS2
            rclpy.init()
            
            # 創建訂閱器
            self.image_subscriber = ROS2CameraSubscriber(
                self.config['image_topic'], 
                self.status_queue
            )
            self.tf_subscriber = ROS2TFSubscriber(
                self.config['tf_topic'],
                self.config['target_frame'],
                self.status_queue
            )
            
            # 啟動 ROS2 處理線程
            self.ros2_running = True
            self.ros2_thread = threading.Thread(target=self._ros2_loop, daemon=True)
            self.ros2_thread.start()
            
            # 啟動數據處理線程
            self.data_thread = threading.Thread(target=self._data_processing_loop, daemon=True)
            self.data_thread.start()
            
            # 启动Web数据更新线程
            self.web_update_thread = threading.Thread(target=self._web_data_update_loop, daemon=True)
            self.web_update_thread.start()
            
            # 更新 UI
            self.ros2_start_btn.config(state=tk.DISABLED)
            self.ros2_stop_btn.config(state=tk.NORMAL)
            self.ros2_status_label.config(text="狀態: 已連接", foreground="green")
            
            # 檢查當前選擇的資料庫是否已連接來啟用存儲功能
            current_db_manager = self._get_current_db_manager()
            if current_db_manager.connected:
                self.storage_start_btn.config(state=tk.NORMAL)
                self.manual_store_btn.config(state=tk.NORMAL)
            else:
                self._add_log(f"💡 提示: 請先連接 {self.current_db_type.upper()} 以啟用存儲功能")
            
            self._add_log("✅ ROS2 連接成功")
            
        except Exception as e:
            self._add_log(f"❌ ROS2 連接失敗: {e}")
            messagebox.showerror("錯誤", f"ROS2 連接失敗:\n{e}")
    
    def _stop_ros2(self):
        """停止 ROS2 連接"""
        if not self.ros2_running:
            return
        
        try:
            self._add_log("🔌 正在斷開 ROS2...")
            
            # 先停止存儲
            if self.storage_running:
                self._stop_storage()
            
            # 停止 ROS2
            self.ros2_running = False
            
            # 清理訂閱器
            if self.image_subscriber:
                self.image_subscriber.destroy_node()
                self.image_subscriber = None
            
            if self.tf_subscriber:
                self.tf_subscriber.destroy_node()
                self.tf_subscriber = None
            
            # 關閉 ROS2
            try:
                rclpy.shutdown()
            except:
                pass
            
            # 更新 UI
            self.ros2_start_btn.config(state=tk.NORMAL)
            self.ros2_stop_btn.config(state=tk.DISABLED)
            self.ros2_status_label.config(text="狀態: 未連接", foreground="red")
            self.storage_start_btn.config(state=tk.DISABLED)
            self.storage_stop_btn.config(state=tk.DISABLED)
            self.manual_store_btn.config(state=tk.DISABLED)
            
            self._add_log("✅ ROS2 連接已斷開")
            
        except Exception as e:
            self._add_log(f"❌ 斷開 ROS2 失敗: {e}")
    
    def _ros2_loop(self):
        """ROS2 處理循環"""
        while self.ros2_running:
            try:
                if self.image_subscriber:
                    rclpy.spin_once(self.image_subscriber, timeout_sec=0.05)
                if self.tf_subscriber:
                    rclpy.spin_once(self.tf_subscriber, timeout_sec=0.05)
                time.sleep(0.01)
            except Exception as e:
                if self.ros2_running:  # 只在運行時報告錯誤
                    self.status_queue.put(('error', f"ROS2 loop error: {e}"))
                break
    
    def _data_processing_loop(self):
        """數據處理循環"""
        while self.ros2_running:
            try:
                # 獲取最新數據並創建同步數據包
                if self.image_subscriber and self.tf_subscriber:
                    image, img_timestamp = self.image_subscriber.get_latest_image()
                    transform, tf_timestamp = self.tf_subscriber.get_latest_transform()
                    
                    if image is not None and transform is not None:
                        # 創建數據包
                        coord_info = self.tf_subscriber.extract_coordinates(transform)
                        unified_timestamp = datetime.now()
                        
                        data_packet = CameraDataPacket(
                            image=image,
                            timestamp=unified_timestamp,
                            position=coord_info['position'],
                            rotation=coord_info['rotation'],
                            coordinate_frame=coord_info['coordinate_frame'],
                            coordinate_method=coord_info['method'],
                            capture_time_iso=unified_timestamp.isoformat(),
                            frame_id=self.status_manager.image_count
                        )
                        
                        # 存儲當前數據包
                        with self.data_lock:
                            self.current_data_packet = data_packet
                
                time.sleep(0.5)  # 2Hz 處理頻率
                
            except Exception as e:
                if self.ros2_running:
                    self.status_queue.put(('error', f"Data processing error: {e}"))
                time.sleep(1.0)
    
    def _web_data_update_loop(self):
        """Web数据更新循环"""
        while self.ros2_running:
            try:
                # 如果Web服务器在运行，更新数据
                if self.web_server_running:
                    self._export_images_for_web()
                
                time.sleep(10.0)  # 每10秒更新一次
                
            except Exception as e:
                if self.ros2_running:
                    self.status_queue.put(('error', f"Web data update error: {e}"))
                time.sleep(10.0)
    
    def _start_storage(self):
        """開始自動存儲"""
        if self.storage_running:
            return
        
        # 檢查當前選擇的資料庫連接狀態
        current_db_manager = self._get_current_db_manager()
        if not current_db_manager.connected:
            messagebox.showwarning("警告", f"{self.current_db_type.upper()} 未連接，請先連接 {self.current_db_type.upper()} 資料庫")
            return
        
        try:
            self.storage_running = True
            self.storage_thread = threading.Thread(target=self._storage_loop, daemon=True)
            self.storage_thread.start()
            
            # 更新 UI
            self.storage_start_btn.config(state=tk.DISABLED)
            self.storage_stop_btn.config(state=tk.NORMAL)
            self.storage_status_label.config(text="狀態: 運行中", foreground="green")
            
            self._add_log(f"💾 自動存儲已開始 (目標: {self.current_db_type.upper()})")
            
        except Exception as e:
            self._add_log(f"❌ 開始存儲失敗: {e}")
    
    def _stop_storage(self):
        """停止自動存儲"""
        if not self.storage_running:
            return
        
        self.storage_running = False
        
        # 更新 UI - 檢查當前資料庫是否連接
        current_db_manager = self._get_current_db_manager()
        self.storage_start_btn.config(state=tk.NORMAL if current_db_manager.connected else tk.DISABLED)
        self.storage_stop_btn.config(state=tk.DISABLED)
        self.storage_status_label.config(text="狀態: 已停止", foreground="orange")
        
        self._add_log("⏹️ 自動存儲已停止")
    
    def _storage_loop(self):
        """存儲循環"""
        last_storage_time = time.time()
        storage_interval = self.config['storage_interval']
        
        while self.storage_running:
            try:
                current_time = time.time()
                if current_time - last_storage_time >= storage_interval:
                    if self._store_current_data():
                        last_storage_time = current_time
                
                time.sleep(1.0)
                
            except Exception as e:
                if self.storage_running:
                    self.status_queue.put(('error', f"Storage loop error: {e}"))
                time.sleep(1.0)
    
    def _manual_store(self):
        """手動存儲當前幀"""
        if self._store_current_data():
            self._add_log("📷 手動存儲完成")
        else:
            self._add_log("❌ 手動存儲失敗")
    
    def _check_services(self):
        """檢查外部服務"""
        def check_in_background():
            # 檢查並初始化 Ollama
            if not self.vlm_analyzer:
                self.vlm_analyzer = OllamaVLMAnalyzer(self.config, self.status_queue)
        
        threading.Thread(target=check_in_background, daemon=True).start()
    
    def _apply_config(self):
        """應用配置"""
        self.config['image_topic'] = self.image_topic_var.get()
        self.config['tf_topic'] = self.tf_topic_var.get()
        self.config['target_frame'] = self.target_frame_var.get()
        self.config['milvus_host'] = self.milvus_host_var.get()
        self.config['milvus_port'] = self.milvus_port_var.get()
        self.config['collection_name'] = self.collection_name_var.get()
        self.config['ollama_url'] = self.ollama_url_var.get()
        self.config['ollama_model'] = self.ollama_model_var.get()
        self.config['enable_ai_analysis'] = self.ai_enable_var.get()
        
        self._add_log("⚙️ 配置已應用")
    
    def _save_config(self):
        """保存配置"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            try:
                self._apply_config()
                with open(filename, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                self._add_log(f"💾 配置已保存到: {filename}")
            except Exception as e:
                self._add_log(f"❌ 保存配置失敗: {e}")
    
    def _load_config(self):
        """載入配置"""
        filename = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    self.config.update(loaded_config)
                
                # 更新 UI
                self.image_topic_var.set(self.config['image_topic'])
                self.tf_topic_var.set(self.config['tf_topic'])
                self.target_frame_var.set(self.config['target_frame'])
                self.milvus_host_var.set(self.config['milvus_host'])
                self.milvus_port_var.set(self.config['milvus_port'])
                self.collection_name_var.set(self.config['collection_name'])
                self.ollama_url_var.set(self.config['ollama_url'])
                self.ollama_model_var.set(self.config['ollama_model'])
                self.ai_enable_var.set(self.config['enable_ai_analysis'])
                
                self._add_log(f"📁 配置已從 {filename} 載入")
            except Exception as e:
                self._add_log(f"❌ 載入配置失敗: {e}")
    
    def on_closing(self):
        """關閉應用程式時的清理"""
        try:
            self._add_log("🛑 正在關閉應用程式...")
            
            # 停止所有運行中的服務
            if self.storage_running:
                self._stop_storage()
            
            if self.ros2_running:
                self._stop_ros2()
            
            # 停止Web服务器
            if self.web_server_running:
                self.web_server_running = False
                if hasattr(self, 'web_server_process') and self.web_server_process:
                    self.web_server_process.terminate()
            
            # 斷開 Milvus
            try:
                connections.disconnect("default")
            except:
                pass
            
            self.master.destroy()
            
        except Exception as e:
            print(f"關閉錯誤: {e}")
            self.master.destroy()

def main():
    """主函數"""
    # 顯示啟動信息
    print("🚀 " + "="*60)
    print("🚀 ROS2 Image Processor GUI Application Starting...")
    print("🚀 " + "="*60)
    print("💡 系統日誌將顯示在此 Terminal 中")
    print("💡 請保持此 Terminal 視窗開啟以查看即時輸出")
    print("🚀 " + "-"*60)
    
    # 檢查依賴
    print("🔍 [Startup] Checking dependencies...")
    if not ROS2_AVAILABLE:
        print("❌ [Startup] ROS2 不可用，請檢查安裝")
        print("💡 [Startup] 請確保已 source ROS2 環境: source /opt/ros/humble/setup.bash")
        return
    else:
        print("✅ [Startup] ROS2 modules available")
    
    if not check_numpy_compatibility():
        print("❌ [Startup] NumPy 版本不兼容，請降級")
        print("💡 [Startup] 執行: pip install 'numpy<2.0,>=1.21.0'")
        return
    else:
        print("✅ [Startup] NumPy version compatible")
    
    print("✅ [Startup] All dependencies checked")
    print("🚀 " + "-"*60)
    
    # 創建 GUI 應用程式
    print("🖥️  [Startup] Creating GUI application...")
    root = tk.Tk()
    app = ROS2ImageProcessorGUI(root)
    
    # 設置關閉處理
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("✅ [Startup] GUI application created successfully")
    print("🚀 " + "="*60)
    print("🎛️ GUI 已啟動！請在圖形界面中操作")
    print("📺 所有系統訊息將在此 Terminal 中顯示")
    print("🚀 " + "="*60)
    
    # 啟動應用程式
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n⚠️ [Shutdown] Received Ctrl+C, shutting down...")
        app.on_closing()
    except Exception as e:
        print(f"❌ [Error] Application error: {e}")
    finally:
        print("👋 [Shutdown] Application closed")

if __name__ == "__main__":
    main()