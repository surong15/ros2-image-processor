# ROS2 Image Processor GUI Application

A fully-featured ROS2 image processing and intelligent analysis system with GUI control interface, AI visual analysis, vector database storage, and web viewer functionality.

## System Features

### Core Functions
- **Real-time Image Processing**: Subscribe to ROS2 camera topics and process image data in real-time
- **Position Synchronization**: Obtain precise position and pose information through TF transforms
- **AI Visual Analysis**: Integrated Ollama LLaVA model for intelligent image description
- **Dual Database Support**: Support for both Milvus and Qdrant vector databases
- **Web Viewer**: Built-in HTTP server providing web interface for image data viewing
- **Graphical Control**: Complete Tkinter GUI supporting real-time monitoring and parameter adjustment

### Technical Highlights
- **Modular Design**: Support runtime database type switching
- **Asynchronous Processing**: Multi-threaded architecture ensures UI responsiveness
- **Status Monitoring**: Real-time display of component connection and operation status
- **Configuration Management**: Support for YAML configuration file saving and loading
- **Error Handling**: Comprehensive exception handling and logging system

## System Requirements

### Basic Environment
- **Operating System**: Ubuntu 20.04/22.04 (Recommended)
- **Python**: 3.8 or above
- **ROS2**: Humble Hawksbill (Recommended) or Foxy Fitzroy
- **Memory**: At least 4GB RAM (8GB+ recommended)

### Core Dependencies
```bash
# ROS2 Related
ros-humble-desktop
ros-humble-cv-bridge
ros-humble-sensor-msgs
ros-humble-geometry-msgs
ros-humble-tf2-msgs

# Python Packages
tkinter
numpy<2.0,>=1.21.0
pillow
requests
pyyaml
```

### Optional Dependencies (Choose based on needs)
```bash
# Milvus Database
pymilvus>=2.3.0

# Qdrant Database  
qdrant-client

# Ollama AI Service
# Requires separate Ollama installation: https://ollama.ai/
```

## Quick Installation

### 1. Environment Preparation
```bash
# Ensure ROS2 environment is set up
source /opt/ros/humble/setup.bash

# Create workspace
mkdir -p ~/ros2_image_processor_ws/src
cd ~/ros2_image_processor_ws/src

# Clone or copy project files
# Place ros2_image_processor.py in this directory
```

### 2. Install Python Dependencies
```bash
# Basic dependencies
pip install numpy==1.24.3 pillow requests pyyaml

# Database dependencies (choose one or install both)
pip install pymilvus>=2.3.0        # For Milvus
pip install qdrant-client           # For Qdrant
```

### 3. Setup Ollama AI Service (Optional)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download LLaVA model
ollama pull llava:latest

# Start Ollama service (default port 11434)
ollama serve
```

### 4. Setup Database (Choose one)

#### Option A: Run Milvus with Docker
```bash
# Download Milvus docker-compose
wget https://github.com/milvus-io/milvus/releases/download/v2.3.2/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
docker-compose up -d

# Check status
docker-compose ps
```

#### Option B: Run Qdrant with Docker
```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

## Usage Guide

### 1. Start System
```bash
# Ensure ROS2 environment is set up
source /opt/ros/humble/setup.bash

# Enter project directory
cd ~/ros2_image_processor_ws/src

# Launch GUI application
python3 ros2_image_processor.py

# Or run in background
python3 ros2_image_processor.py &
```

### 2. GUI Operation Workflow

#### Step 1: Connect External Services
1. **Database Connection**:
   - Select database type (Milvus/Qdrant) on "Control Panel" page
   - Click "Connect Database"
   - Confirm connection status shows "Connected"

2. **AI Service Connection**:
   - Click "Check Ollama"
   - Confirm status shows "Connected"

#### Step 2: Configure ROS2 Topics
1. Switch to "Configuration Settings" page
2. Modify the following parameters to match your robot:
   ```yaml
   image_topic: '/your_robot/camera/image_raw'    # Camera topic
   tf_topic: '/your_robot/tf'                     # TF topic  
   target_frame: 'your_robot_base_frame'          # Target coordinate frame
   ```
3. Click "Apply Configuration"

#### Step 3: Start Data Collection
1. Return to "Control Panel"
2. Click "Connect ROS2"
3. Wait for status to show "Connected"
4. Click "Start Storage" to begin automatic data collection

#### Step 4: Monitor and View
1. **Real-time Monitoring**: Switch to "Status Monitor" to view statistics
2. **Web Viewer**: Click "Open Web Viewer" to view historical images
3. **Manual Storage**: Use "Manually Store Current Frame" button

### 3. Web Viewer Usage
- Browser access: `http://localhost:8889`
- Automatically displays the most recent 20 images
- Includes position information, AI analysis results, etc.
- Auto-refreshes data every 30 seconds

## 🔧 Customization Configuration Guide

### 1. Modify Robot-Related Configuration

Based on your robot system, modify the following configuration:

```python
# Modify in _load_default_config() function
def _load_default_config(self):
    return {
        # === Required Modifications ===
        'image_topic': '/YOUR_ROBOT_NAME/camera/image_raw',    # Change to your camera topic
        'tf_topic': '/YOUR_ROBOT_NAME/tf',                     # Change to your TF topic
        'target_frame': 'YOUR_ROBOT_BASE_FRAME',               # Change to your base coordinate frame
        
        # === Optional Modifications ===
        'collection_name': 'YOUR_PROJECT_images',             # Custom dataset name
        'storage_interval': 5.0,                              # Auto storage interval (seconds)
        'web_viewer_port': 8889,                              # Web viewer port
        
        # Other configurations can remain default...
    }
```

### 2. Customize AI Analysis Prompts

Modify the analysis prompt in the `OllamaVLMAnalyzer` class:

```python
# Modify prompt in analyze_image method
payload = {
    "model": self.ollama_model,
    "prompt": "Describe this image for robot navigation purposes, focusing on visible obstacles, landmarks, and environmental features.",  # Custom prompt
    "images": [img_base64],
    "stream": False,
    "options": {"temperature": 0.3}
}
```

## Monitoring and Debugging

### Performance Tuning

#### Adjust Processing Frequency:
```python
# Modify sleep time in _data_processing_loop
time.sleep(0.5)  # 2Hz processing frequency, can adjust to 0.1 (10Hz) or 1.0 (1Hz)
```

#### Adjust Storage Interval:
```python
# Modify in configuration
'storage_interval': 3.0,  # Store every 3 seconds, adjust based on needs
```

#### Optimize Vector Dimensions:
```python
# Adjust based on storage space and retrieval accuracy needs
'vector_dim': 256,  # Reduce to 256 to save space, or increase to 1024 for better accuracy
```

## Troubleshooting

### Common Issues and Solutions

#### 1. ROS2 Connection Failure
```bash
# Check ROS2 environment
echo $ROS_DOMAIN_ID
ros2 topic list

# Ensure environment variables are correctly set
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
```

#### 2. Camera Topic Not Found
```bash
# List all available topics
ros2 topic list | grep image

# Check topic type
ros2 topic info /your_camera_topic

# Confirm image data format
ros2 topic echo /your_camera_topic --no-arr
```

#### 3. TF Coordinate Frame Issues
```bash
# View TF tree structure
ros2 run tf2_tools view_frames.py

# Monitor TF in real-time
ros2 run tf2_ros tf2_echo source_frame target_frame

# List all coordinate frames
ros2 topic echo /tf --no-arr
```

#### 4. Database Connection Issues
```bash
# Milvus connection check
docker ps | grep milvus
telnet localhost 19530

# Qdrant connection check  
docker ps | grep qdrant
curl http://localhost:6333/collections
```

#### 5. Ollama AI Service Issues
```bash
# Check Ollama service status
curl http://localhost:11434/api/tags

# Check installed models
ollama list

# Re-pull model
ollama pull llava:latest
```

#### 6. NumPy Version Compatibility Issues
```bash
# Check NumPy version
python3 -c "import numpy; print(numpy.__version__)"

# If version ≥2.0, downgrade required
pip install 'numpy<2.0,>=1.21.0'
```

### Debugging Tips

#### Step-by-Step Testing:
1. First test ROS2 connection
2. Then test database connection
3. Next test AI service
4. Finally test complete workflow


## Default Program Names

This program uses default values for certain parameters. If not modified in the configuration file or GUI, the following defaults will be used:

| Parameter            | Default Value                        | Description                     |
|----------------------|--------------------------------------|---------------------------------|
| Image Topic          | `/ROBOTNAME/camera/image_raw`       | ROS2 camera topic               |
| TF Topic             | `/ROBOTNAME/tf`                     | ROS2 TF coordinate topic        |
| Target Frame         | `tn__7R05D00002_only_bottom_sim_`   | Robot base coordinate frame     |
| Collection Name      | `ros2_camera_images`                | Vector database collection name |
| Milvus Host          | `localhost`                         | Milvus server host              |
| Milvus Port          | `19530`                             | Milvus server port              |
| Qdrant Host          | `localhost`                         | Qdrant server host              |
| Qdrant Port          | `6333`                              | Qdrant server port              |
| Web Viewer Port      | `8889`                              | Web viewer default port         |
| AI Model Name        | `llava:latest`                      | Ollama LLaVA model              |

To customize, modify these parameters in the GUI or `config.yaml`.
