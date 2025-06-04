# Angular Development Dockerfile
FROM node:20-alpine

# Set working directory
WORKDIR /app

# Install global Angular CLI
RUN npm install -g @angular/cli@latest

# Copy package files first for better caching
COPY package*.json ./

# Install all dependencies (including devDependencies)
RUN npm install

# Copy source code
COPY . .

# Expose Angular dev server port
EXPOSE 4200

# Start Angular development server
CMD ["ng", "serve", "--host", "0.0.0.0", "--port", "4200"]