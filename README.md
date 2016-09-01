# leuko-tf

### local development
* Create a virtual machine using the latest version of Ubuntu 64-bit.

* Launch the virtual machine, upgrade and update
```
sudo apt upgrade
sudo apt update
```

* Install TensorFlow dependencies
```
sudo apt install python-pip python-dev python-virtualenv git
```

* Create virtual environment for TensorFlow development
```
virtualenv --system-site-packages ~/tensorflow
```

* Active the virtual environment
```
source ~/tensorflow/bin/activate
```

* Install TensorFlow (see the TensorFlow installation guide for the latest Linux 64-bit, CPU only, Python 2.7 whl file)
```
pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
```

* Clone this repository if you haven't already
```
git clone https://github.com/boerjames/leuko-tf.git
```

Use your favority IDE (mine is PyCharm) to work on this project using the virtual environment we created.

### server deployment
