# Face_Recognizer-Image-
This repository contains the code for a simple face recognition model using OpenCV

This Mini project uses OpenCV to detect faces based on the trained data



You'll find two python files
--> face_recognition
--> Face_recognizer



You'll also find one xml file
--> haar_face (haarcascade classifier)



The first python file trains a opencv on the given pictures of three people as you'll see in the file.


The 'directory' in the first file refers to the directory of the 'TRAIN AND VALIDATION DATA' data that is given here in the repository.


And the last part of this file contains code to simply save the trained model so we can simple load it in the 'Face_recognition' file


Now in the second python file is where the image identification is done and thus it uses the data in the 'VALID' folder in the 'TRAIN AND VALIDATION DATA' folder
.So, the path in 'img.imread()' function refers to an image in the 'VALID' folder.


The people list in both python file is hard coded here... you may use the os.listdir function to loop over the folder names if u want to do it the other way...


This model can be trained on basis on facial recognition and one can use more data to train.


The data given in the 'TRAIN AND VALIDATION DATA' is a simple sample

Both the python file uses harcascade classifier... this is one of the bad classifier you can choose but not the worst there are better ones... but this what i used

