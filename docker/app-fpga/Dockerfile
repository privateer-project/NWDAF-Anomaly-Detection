FROM aimilefth/privateer-ad:app

ENV DEBIAN_FRONTEND=noninteractive
ARG XRT_GIT_URL=https://github.com/Xilinx/XRT.git
ARG XRT_GIT_VERSION=202210.2.13.466
ARG XRT_SRC_DIR="/tmp/XRT_SRC"

# APT Installs
RUN apt update && \
    apt install curl git pkg-config cmake -y

# Clone XRT source code
RUN echo "Cloning XRT from ${XRT_GIT_URL} tag ${XRT_GIT_VERSION} into ${XRT_SRC_DIR}" && \
    git clone --branch ${XRT_GIT_VERSION} --depth 1 ${XRT_GIT_URL} ${XRT_SRC_DIR}
      
# Build XRT, Package, and Install
RUN echo "Running xrtdeps.sh..." && \
    cd "${XRT_SRC_DIR}/src/runtime_src/tools/scripts/" && \
    ./xrtdeps.sh -docker

# # For debugging - to confirm the path pybind11 --cmakedir would give
# RUN echo "Attempting to find pybind11 CMake directory..." && \
# (python -m pybind11 --cmakedir || echo "pybind11 module not found or --cmakedir failed")

# Use system cmake only for the build step
RUN PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH" \
    && echo "Configuring and Building XRT (make)..." \
    && cd ${XRT_SRC_DIR}/build \
    && env CXXFLAGS="${CXXFLAGS} -Wno-error=deprecated-declarations" \
       pybind11_DIR="/opt/conda/lib/python3.11/site-packages/pybind11/share/cmake/pybind11" \
       ./build.sh -opt -noctest

RUN echo "XRT packaging completed. Now installing .deb..." && \
    cd ${XRT_SRC_DIR}/build/Release && \
    ls && \
    echo "Installing:  $(ls -1 xrt_*-xrt.deb)" && \
    apt-get install -y ./xrt_*-xrt.deb && \
    echo "XRT installation completed."

# Clean up XRT source and build directories
RUN rm -rf ${XRT_SRC_DIR}

# Setup
ARG SETUP_DIR_ARG=/setup_dir
ARG SETUP_FILE_ARG=setup.sh
ENV SETUP_DIR=${SETUP_DIR_ARG}
ENV SETUP_FILE=${SETUP_FILE_ARG}

# Create setup directory and copy setup file
RUN mkdir -p ${SETUP_DIR_ARG}
COPY ${SETUP_FILE} ${SETUP_DIR_ARG}

# Modify permissions and ownership of setup file and working directory
RUN chmod u+x ${SETUP_DIR_ARG}/${SETUP_FILE_ARG}

# Copy requirements.txt to the container
COPY ../fpga-requirements.txt /tmp/fpga-requirements.txt
# Use uv to install Python dependencies
RUN uv pip install -r /tmp/fpga-requirements.txt --system --no-cache --compile-bytecode && rm /tmp/fpga-requirements.txt
# Marimo config. 
COPY .marimo.toml /root/

# Set PYTHONPATH to ensure /app is on sys.path
ENV PYTHONPATH=/app

EXPOSE 8080

CMD bash -c "source ${SETUP_DIR}/${SETUP_FILE} && marimo edit --host '0.0.0.0' --port '8080' /app/privateer_ad/marimo/marimo.py"