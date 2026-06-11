FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python packages (CPU version of PyTorch for low-cost, high-compatibility execution)
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir -r requirements.txt

# Create a symlink to python for server.js compatibility
RUN mkdir -p .venv/bin && ln -sf $(which python3) .venv/bin/python

# Copy package.json and install Node dependencies
COPY package.json .
RUN npm install

# Copy application files
COPY . .

# Expose port
EXPOSE 8500

# Run the app
CMD ["node", "server.js"]
