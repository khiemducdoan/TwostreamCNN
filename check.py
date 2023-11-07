import os
folderfather = 'asl_alphabet_train\\asl_alphabet_train'

for folder in os.listdir(folderfather):
    f = open("demofile2.txt", "r")
    os.makedirs(os.path.join('asl_alphabet_test\\asl_alphabet_test',folder ))
