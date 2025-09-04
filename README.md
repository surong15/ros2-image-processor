# ROS2 Image Processor GUI Application

一個功能完整的ROS2圖像處理與智能分析系統，具備GUI控制界面、AI視覺分析、向量數據庫存儲和Web查看器功能。

![System Architecture](https://img.shields.io/badge/ROS2-Humble-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![AI](https://img.shields.io/badge/AI-LLaVA-orange)
![Database](https://img.shields.io/badge/VectorDB-Milvus%2FQdrant-purple)

## 🌟 系統特色

### 🎯 核心功能
- **實時圖像處理**: 訂閱ROS2攝像頭話題，實時處理圖像數據
- **位置同步**: 通過TF變換獲取精確的位置和姿態信息
- **AI視覺分析**: 集成Ollama LLaVA模型進行智能圖像描述
- **雙數據庫支持**: 支持Milvus和Qdrant向量數據庫存儲
- **Web查看器**: 內置HTTP服務器，提供圖像數據的Web界面查看
- **圖形化控制**: 完整的Tkinter GUI，支持實時監控和參數調整

### 🔧 技術亮點
- **模組化設計**: 支持運行時切換數據庫類型
- **異步處理**: 多線程架構確保UI響應性
- **狀態監控**: 實時顯示各組件連接和運行狀態
- **配置管理**: 支持YAML配置文件的保存和加載
- **錯誤處理**: 完善的異常處理和日誌記錄系統

## 📋 系統要求

### 基礎環境
- **操作系統**: Ubuntu 20.04/22.04 (推薦)
- **Python**: 3.8 以上版本
- **ROS2**: Humble Hawksbill (推薦) 或 Foxy Fitzroy
- **記憶體**: 至少 4GB RAM (推薦 8GB+)

### 核心依賴
```bash
# ROS2相關
ros-humble-desktop
ros-humble-cv-bridge
ros-humble-sensor-msgs
ros-humble-geometry-msgs
ros-humble-tf2-msgs

# Python套件
tkinter
numpy<2.0,>=1.21.0
pillow
requests
pyyaml
```

### 可選依賴 (根據需求選擇)
```bash
# Milvus數據庫
pymilvus>=2.3.0

# Qdrant數據庫  
qdrant-client

# Ollama AI服務
# 需要單獨安裝Ollama: https://ollama.ai/
```

## ⚡ 快速安裝

### 1. 環境準備
```bash
# 確保ROS2環境已設置
source /opt/ros/humble/setup.bash

# 創建工作空間
mkdir -p ~/ros2_image_processor_ws/src
cd ~/ros2_image_processor_ws/src

# 克隆或複製項目文件
# 將ros2_image_processor.py放入此目錄
```

### 2. 安裝Python依賴
```bash
# 基礎依賴
pip install numpy==1.24.3 pillow requests pyyaml

# 數據庫依賴 (選擇其一或兩者都安裝)
pip install pymilvus>=2.3.0        # 用於Milvus
pip install qdrant-client           # 用於Qdrant
```

### 3. 設置Ollama AI服務 (可選)
```bash
# 安裝Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下載LLaVA模型
ollama pull llava:latest

# 啟動Ollama服務 (預設端口11434)
ollama serve
```

### 4. 設置數據庫 (選擇其一)

#### 選項A: 使用Docker運行Milvus
```bash
# 下載Milvus docker-compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.2/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 啟動Milvus
docker-compose up -d

# 檢查狀態
docker-compose ps
```

#### 選項B: 使用Docker運行Qdrant
```bash
# 啟動Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

## 🚀 使用指南

### 1. 啟動系統
```bash
# 確保ROS2環境已設置
source /opt/ros/humble/setup.bash

# 進入項目目錄
cd ~/ros2_image_processor_ws/src

# 啟動GUI應用程序
python3 ros2_image_processor.py

# 或者在後台運行
python3 ros2_image_processor.py &
```

### 2. GUI操作流程

#### 步驟1: 連接外部服務
1. **數據庫連接**:
   - 在"控制面板"頁面選擇數據庫類型 (Milvus/Qdrant)
   - 點擊"🗃️ 連接資料庫"
   - 確認連接狀態顯示為"已連接"

2. **AI服務連接**:
   - 點擊"🤖 檢查 Ollama"
   - 確認狀態顯示為"已連接"

#### 步驟2: 配置ROS2話題
1. 切換到"⚙️ 配置設定"頁面
2. 修改以下參數以適應您的機器人:
   ```yaml
   image_topic: '/your_robot/camera/image_raw'    # 攝像頭話題
   tf_topic: '/your_robot/tf'                     # TF話題  
   target_frame: 'your_robot_base_frame'          # 目標坐標框架
   ```
3. 點擊"🔄 應用配置"

#### 步驟3: 啟動數據收集
1. 返回"🎛️ 控制面板"
2. 點擊"🔗 連接 ROS2"
3. 等待狀態顯示"已連接"
4. 點擊"💾 開始存儲"開始自動數據收集

#### 步驟4: 監控和查看
1. **實時監控**: 切換到"📊 狀態監控"查看統計信息
2. **Web查看器**: 點擊"🌐 開啟 Web 查看器"查看歷史圖像
3. **手動存儲**: 使用"📷 手動存儲當前幀"按鈕

### 3. Web查看器使用
- 瀏覽器訪問: `http://localhost:8889`
- 自動顯示最近的20張圖像
- 包含位置信息、AI分析結果等
- 每30秒自動刷新數據

## 🔧 個人化配置指南

### 1. 修改機器人相關配置

根據您的機器人系統，需要修改以下配置:

```python
# 在_load_default_config()函數中修改
def _load_default_config(self):
    return {
        # === 必須修改的項目 ===
        'image_topic': '/YOUR_ROBOT_NAME/camera/image_raw',    # 改為您的相機話題
        'tf_topic': '/YOUR_ROBOT_NAME/tf',                     # 改為您的TF話題
        'target_frame': 'YOUR_ROBOT_BASE_FRAME',               # 改為您的基座坐標系
        
        # === 可選修改項目 ===
        'collection_name': 'YOUR_PROJECT_images',             # 自定義數據集名稱
        'storage_interval': 5.0,                              # 自動存儲間隔(秒)
        'web_viewer_port': 8889,                              # Web查看器端口
        
        # 其他配置保持預設即可...
    }
```

### 2. 自定義AI分析提示詞

修改`OllamaVLMAnalyzer`類中的分析提示:

```python
# 在analyze_image方法中修改prompt
payload = {
    "model": self.ollama_model,
    "prompt": "請針對機器人導航需求描述這張圖片，重點說明可見的障礙物、地標和環境特徵。",  # 自定義提示
    "images": [img_base64],
    "stream": False,
    "options": {"temperature": 0.3}
}
```

### 3. 調整數據庫存儲內容

根據需求修改存儲的數據字段:

```python
# 在store_data_packet方法中添加自定義字段
metadata = json.dumps({
    "capture_method": "ros2_gui_application",
    "coordinate_method": data_packet.coordinate_method,
    "coordinate_frame": data_packet.coordinate_frame,
    "image_shape": list(data_packet.image.shape),
    "frame_id": data_packet.frame_id,
    "app_version": "gui_v1.0",
    # === 添加您的自定義字段 ===
    "robot_name": "YOUR_ROBOT_NAME",
    "mission_id": "YOUR_MISSION_ID",
    "location_tags": ["indoor", "warehouse"],  # 自定義標籤
})
```

### 4. 修改圖像處理參數

調整圖像壓縮和特徵提取:

```python
# 在_compress_image方法中修改
def _compress_image(self, image: np.ndarray, quality=85) -> str:  # 提高圖像質量
    # 修改最大尺寸限制
    if max(pil_image.size) > 1024:  # 增加到1024像素
        ratio = 1024 / max(pil_image.size)
        new_size = tuple(int(dim * ratio) for dim in pil_image.size)
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
```

### 5. 自定義GUI界面

修改界面標題和布局:

```python
# 在__init__方法中修改
self.master.title("YOUR_PROJECT_NAME - ROS2 Image Processor")
self.master.geometry("1200x800")  # 調整窗口大小

# 在_create_gui方法中修改標題
title_label = ttk.Label(main_frame, text="🤖 YOUR_PROJECT_NAME Image Processor", 
                       font=('Arial', 16, 'bold'))
```

### 6. 添加自定義功能

#### 添加新的數據處理功能:

```python
def _custom_data_processing(self, image, position):
    """自定義數據處理邏輯"""
    try:
        # 例如: 添加圖像預處理
        processed_image = self._apply_custom_filter(image)
        
        # 例如: 添加位置驗證
        if self._validate_position(position):
            return processed_image, True
        else:
            return image, False
            
    except Exception as e:
        self._add_log(f"自定義處理錯誤: {e}")
        return image, False

def _apply_custom_filter(self, image):
    """應用自定義圖像濾鏡"""
    # 添加您的圖像處理邏輯
    return image

def _validate_position(self, position):
    """驗證位置數據有效性"""
    # 添加您的位置驗證邏輯
    return True
```

## 📊 監控和調試

### 1. 日誌系統
所有系統日誌都會在終端實時顯示，包括:
- 🔗 連接狀態
- 📷 圖像接收
- 📡 TF數據更新  
- 💾 數據存儲
- 🤖 AI分析結果
- ⚠️ 錯誤信息

### 2. 狀態監控
GUI中的"📊 狀態監控"頁面提供:
- 接收圖像數量統計
- TF數據更新頻率
- 數據存儲成功率
- 各服務連接狀態
- 當前數據包詳細信息

### 3. 性能調優

#### 調整處理頻率:
```python
# 在_data_processing_loop中修改睡眠時間
time.sleep(0.5)  # 2Hz處理頻率，可調整為0.1 (10Hz)或1.0 (1Hz)
```

#### 調整存儲間隔:
```python
# 在配置中修改
'storage_interval': 3.0,  # 每3秒存儲一次，可根據需求調整
```

#### 優化向量維度:
```python
# 根據存儲空間和檢索精度需求調整
'vector_dim': 256,  # 減小到256以節省空間，或增加到1024以提高精度
```

## 🔍 故障排除

### 常見問題及解決方案

#### 1. ROS2連接失敗
```bash
# 檢查ROS2環境
echo $ROS_DOMAIN_ID
ros2 topic list

# 確保環境變量正確設置
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
```

#### 2. 找不到攝像頭話題
```bash
# 列出所有可用話題
ros2 topic list | grep image

# 檢查話題類型
ros2 topic info /your_camera_topic

# 確認圖像數據格式
ros2 topic echo /your_camera_topic --no-arr
```

#### 3. TF坐標框架問題
```bash
# 查看TF樹結構
ros2 run tf2_tools view_frames.py

# 實時監控TF
ros2 run tf2_ros tf2_echo source_frame target_frame

# 列出所有坐標框架
ros2 topic echo /tf --no-arr
```

#### 4. 數據庫連接問題
```bash
# Milvus連接檢查
docker ps | grep milvus
telnet localhost 19530

# Qdrant連接檢查  
docker ps | grep qdrant
curl http://localhost:6333/collections
```

#### 5. Ollama AI服務問題
```bash
# 檢查Ollama服務狀態
curl http://localhost:11434/api/tags

# 檢查已安裝模型
ollama list

# 重新拉取模型
ollama pull llava:latest
```

#### 6. NumPy版本兼容性問題
```bash
# 檢查NumPy版本
python3 -c "import numpy; print(numpy.__version__)"

# 如果版本≥2.0，需要降級
pip install 'numpy<2.0,>=1.21.0'
```

### 調試技巧

#### 啟用詳細日誌:
修改代碼中的日誌級別以獲得更多調試信息。

#### 使用測試模式:
```python
# 在__init__方法中添加測試標誌
self.test_mode = True  # 啟用測試模式

# 在相應方法中添加測試數據
if self.test_mode:
    self._generate_test_data()
```

#### 分步驟測試:
1. 先測試ROS2連接
2. 再測試數據庫連接
3. 然後測試AI服務
4. 最後測試完整流程


## 🏷️ 程式預設名稱說明

本程式部分參數有預設值，若未在設定檔或GUI中修改，將採用以下預設：

| 參數            | 預設值                          | 說明                     |
|-----------------|----------------------------------|--------------------------|
| 圖像話題        | `/ROBOTNAME/camera/image_raw`       | ROS2攝像頭話題           |
| TF話題          | `/ROBOTNAME/tf`                     | ROS2 TF座標話題          |
| 目標座標框架    | `tn__7R05D00002_only_bottom_sim_`| 機器人基座座標系         |
| Collection Name | `ros2_camera_images`             | 向量資料庫集合名稱       |
| Milvus Host     | `localhost`                      | Milvus伺服器主機         |
| Milvus Port     | `19530`                          | Milvus伺服器端口         |
| Qdrant Host     | `localhost`                      | Qdrant伺服器主機         |
| Qdrant Port     | `6333`                           | Qdrant伺服器端口         |
| Web Viewer Port | `8889`                           | Web查看器預設端口        |
| AI模型名稱      | `llava:latest`                   | Ollama LLaVA模型         |

如需自訂，請於GUI或`config.yaml`中修改上述參數。

---

## 📝 配置文件範例

創建 `config.yaml` 文件:

```yaml
# ROS2設定
image_topic: '/ROBOTNAME/camera/image_raw'
tf_topic: '/ROBOTNAME/tf'  
target_frame: 'tn__7R05D00002_only_bottom_sim_'

# 數據庫設定
database_type: 'milvus'  # 或 'qdrant'
milvus_host: 'localhost'
milvus_port: '19530'
qdrant_host: 'localhost'
qdrant_port: '6333'
collection_name: 'ros2_camera_images'

# AI分析設定
ollama_url: 'http://localhost:11434'
ollama_model: 'llava:latest'
enable_ai_analysis: true
ai_timeout: 30

# 存儲設定
storage_interval: 5.0
vector_dim: 512
web_viewer_port: 8889

# 處理設定
processing_frequency: 2.0
```

## 🚀 進階應用

### 1. 批量數據處理
使用腳本模式處理大量歷史數據:

```python
# 創建batch_processor.py
from ros2_image_processor import ROS2ImageProcessorGUI

def batch_process():
    # 實現批量處理邏輯
    pass
```

### 2. 遠程監控
配置遠程訪問Web界面:

```python
# 修改Web服務器綁定地址
httpd = socketserver.TCPServer(("0.0.0.0", self.web_server_port), CustomHandler)
```

### 3. 數據導出
添加數據導出功能:

```python
def export_data_to_csv(self):
    """導出數據到CSV文件"""
    # 實現CSV導出邏輯
    pass
```

## 📄 授權

本項目採用 MIT 授權條款。

## 🤝 貢獻指南

歡迎提交Issue和Pull Request！

1. Fork 此項目
2. 創建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 開啟 Pull Request

## 📞 技術支援

如遇到問題，請提供以下信息:
- 操作系統版本
- ROS2版本
- Python版本
- 完整錯誤日誌
- 系統配置信息

---

**注意**: 本系統設計用於機器人數據收集和分析，請確保在安全環境中部署和測試。
