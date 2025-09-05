FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel

WORKDIR /app
ENV PYTHONPATH=/app

# Install uv for package management
RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache --compile-bytecode .

CMD ["/bin/bash"]