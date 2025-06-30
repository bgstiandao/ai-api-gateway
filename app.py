from flask import Flask, request, jsonify, render_template
import random
import base64
import os
import logging
import threading
import time
from PIL import Image
from io import BytesIO
from transformers import pipeline
from flasgger import Swagger

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SWAGGER'] = {'title': 'AI API Factory'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB 限制
Swagger(app)

# 模型全局变量
models_loaded = False
image_captioner = None
sentiment_analyzer = None
model_loading_started = False

def load_models_in_background():
    """在后台线程加载模型"""
    global models_loaded, image_captioner, sentiment_analyzer, model_loading_started
    
    if model_loading_started:
        return
        
    model_loading_started = True
    
    # 使用 Hugging Face Spaces 推荐的缓存目录
    cache_dir = "/cache/huggingface"
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    logger.info(f"使用缓存目录: {cache_dir}")
    
    # 确保缓存目录存在
    try:
        if not os.path.exists(cache_dir):
            logger.info(f"尝试创建缓存目录: {cache_dir}")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"缓存目录已创建: {cache_dir}")
        else:
            logger.info(f"缓存目录已存在: {cache_dir}")
    except Exception as e:
        logger.error(f"处理缓存目录失败: {str(e)}")
        return
    
    logger.info("开始后台加载模型...")
    
    try:
        # 加载情感分析模型
        logger.info("加载情感分析模型...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        logger.info("情感分析模型加载完成")
        
        # 加载图像描述模型
        logger.info("加载图像描述模型...")
        image_captioner = pipeline(
            "image-to-text", 
            model="nlpconnect/vit-gpt2-image-captioning"
        )
        logger.info("图像描述模型加载完成")
        
        models_loaded = True
        logger.info("所有模型加载完毕")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        # 重试机制
        time.sleep(5)
        logger.info("尝试重新加载模型...")
        load_models_in_background()

# 应用启动时开始加载模型
threading.Thread(target=load_models_in_background, daemon=True).start()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html', models_ready=models_loaded)

@app.route('/health')
def health_check():
    """健康检查端点"""
    status = "服务运行中" if models_loaded else "模型加载中..."
    return jsonify({
        "status": status,
        "models_ready": models_loaded
    })

@app.route('/random', methods=['POST'])
def get_random():
    """
    生成随机数API
    ---
    tags:
      - 实用工具
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - max_value
          properties:
            max_value:
              type: integer
              description: 随机数的最大值
              example: 100
    responses:
      200:
        description: 成功生成随机数
        schema:
          type: object
          properties:
            random_number:
              type: integer
              description: 生成的随机数
      400:
        description: 缺少参数或参数无效
    """
    if not request.json or 'max_value' not in request.json:
        return jsonify({'error': 'Missing max_value parameter'}), 400
    max_val = request.json['max_value']
    return jsonify({'random_number': random.randint(0, max_val)})

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    """
    情绪分析API
    ---
    tags:
      - AI分析
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
    responses:
      200:
        description: 分析结果
        schema:
          type: object
          properties:
            sentiment:
              type: string
      503:
        description: 模型未就绪
    """
    if not models_loaded:
        return jsonify({'error': '模型仍在加载中，请稍后重试'}), 503
    
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'Missing text parameter'}), 400
    
    text = request.json['text']
    try:
        result = sentiment_analyzer(text)[0]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'情感分析失败: {str(e)}'}), 500

@app.route('/caption', methods=['POST'])
def image_caption():
    """
    图像描述生成API
    ---
    tags:
      - AI分析
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - image
          properties:
            image:
              type: string
              format: base64
              description: base64编码的图像数据
              example: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    responses:
      200:
        description: 图像描述结果
        schema:
          type: object
          properties:
            generated_text:
              type: string
              description: 生成的图像描述
      400:
        description: 缺少图像数据
      503:
        description: 模型未就绪
    """
    if not models_loaded:
        return jsonify({'error': '模型仍在加载中，请稍后重试'}), 503
    
    if not request.json or 'image' not in request.json:
        return jsonify({'error': 'Missing image data'}), 400
    
    try:
        img_data = base64.b64decode(request.json['image'])
        image = Image.open(BytesIO(img_data))
        
        # 检查图像模式并转换为RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        result = image_captioner(image)[0]
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'图像处理失败: {str(e)}'}), 500

@app.errorhandler(400)
def bad_request(e):
    return jsonify(error="无效请求"), 400

@app.errorhandler(500)
def server_error(e):
    return jsonify(error="内部服务器错误"), 500

@app.errorhandler(503)
def service_unavailable(e):
    return jsonify(error="服务暂时不可用，模型加载中"), 503

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)