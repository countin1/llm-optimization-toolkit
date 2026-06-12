FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置入口
ENTRYPOINT ["python", "-m", "loraforge"]
CMD ["data", "--augment", "3"]
