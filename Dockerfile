# The MUSE code works with just pytorch 0.4.1
# You can choose different label for cuda and cudnn depending on that on your system
FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime

LABEL maintainer "Siddharth Yadav <sedflix@gmail.com>"

RUN apt-get update && conda install -y jupyter scipy matplotlib scikit-learn faiss-gpu -c pytorch && conda clean --packages --yes

CMD ["bash"]

