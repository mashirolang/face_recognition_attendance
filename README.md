# face_recognition_attendance

### Todo

- [ ] Cleaning the code by placing most of the reapeted things into functions in `utils.py`

### In Progress

- [ ] Adding a database(preferably local)
- [ ] Integrating it on Internet of Things(IOT) such as GSM module

### Done ✓

- [x] add_data.py
- [x] train_model.py
- [x] Testing and optimizing (With over 1030 different face datasets from kaggle.com)
- [x] Adding a function that will utilize the Model in order to recognize students faces(OPENCV)
- [x] Change of plan: from OPENCV to DEEPFACE, this will make the work easier and more accurately

### Summary

- `add_data.py` - It is used to register new faces into datasets this works but manually taking a picture
- `add_data_retina.py` - It is used to register new faces into datasets but with retina library to accurately detect faces
- `train_model.py` - It is used to train the pkl model which will be used on deepface
- `face_recognition.py` - Title explains it, self explanatory. This is the one who do all the work(probably will be change into `main.py`)
