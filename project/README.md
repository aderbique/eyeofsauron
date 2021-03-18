# Project

### Quick Start
1. Change to the project directory `cd project`.
2. For debian based systems, run `./set_environment.sh` to install python3.8, pip, and install required pip packages for project.\
3. Activate the virtual environment by running `source venv/bin/activate`

### Steps for running on WSL with Graphics
Follow instructions to download [vsXsrv](https://www.ctrl.blog/entry/how-to-x-on-wsl.html) and run. Remember to `export DISPLAY=:0`
Follow instructions to develop WSL Files. [Link Here](https://code.visualstudio.com/docs/remote/wsl)

## Task One
```
(venv) project$ python task1/main.py
```
### Example Output
```
(venv) aderbique@ADERBIQUE-W10:~/academics/eyeofsauron/project$ python task1/main.py
...
Epoch 1/10
1691/1691 [==============================] - 2s 813us/step - loss: 0.5826 - accuracy: 0.8406
Epoch 2/10
1691/1691 [==============================] - 1s 815us/step - loss: 0.2736 - accuracy: 0.8933
Epoch 3/10
1691/1691 [==============================] - 1s 826us/step - loss: 0.2539 - accuracy: 0.8998
Epoch 4/10
1691/1691 [==============================] - 2s 939us/step - loss: 0.2544 - accuracy: 0.9108
Epoch 5/10
1691/1691 [==============================] - 1s 831us/step - loss: 0.1922 - accuracy: 0.9323
Epoch 6/10
1691/1691 [==============================] - 1s 842us/step - loss: 0.1804 - accuracy: 0.9350
Epoch 7/10
1691/1691 [==============================] - 1s 819us/step - loss: 0.1804 - accuracy: 0.9291
Epoch 8/10
1691/1691 [==============================] - 1s 851us/step - loss: 0.1904 - accuracy: 0.9271
Epoch 9/10
1691/1691 [==============================] - 1s 768us/step - loss: 0.1398 - accuracy: 0.9369
Epoch 10/10
1691/1691 [==============================] - 1s 746us/step - loss: 0.1312 - accuracy: 0.9458
529/529 [==============================] - 0s 553us/step - loss: 0.1286 - accuracy: 0.9373
Print the loss and accuracy of the model on the dataset
Loss [0,1]: 0.1286 Accuracy [0,1]: 0.9373
```

## Task Three
```
(venv) project$ python task3/main.py
```