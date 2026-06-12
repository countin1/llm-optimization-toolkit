FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置入口
ENTRYPOINT ["python", "-m", "promptforge"]
CMD ["search", "--method", "grid", "--max-questions", "5"]
