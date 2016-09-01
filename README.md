# leuko-tf

### local development
1. Create a virtual machine using the latest version of Ubuntu 64-bit.
2. Launch the virtual machine, upgrade and update
```
sudo apt upgrade
sudo apt update
```
3. Install TensorFlow dependencies
```
sudo apt install python-pip python-dev python-virtualenv git
```
4. Create virtual environment for TensorFlow development
```
virtualenv --system-site-packages ~/tensorflow
```
5. Active the virtual environment
```
source ~/tensorflow/bin/activate
```
6. Install TensorFlow (see the TensorFlow installation guide for the latest Linux 64-bit, CPU only, Python 2.7 whl file)
```
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
```
7. Clone this repository if you haven't already
```
git clone https://github.com/boerjames/leuko-tf.git
```
8. Use your favority IDE (mine is PyCharm) to work on this project using the virtual environment we created.

### server deployment
