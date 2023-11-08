import os

import shutil

parentFolder = 'asl_alphabet_train\\asl_alphabet_train'
streamData = 'streamData'
streamA = 'streamA'
streamB = 'streamB'

if os.path.isdir(os.path.join(streamData,streamA)) & os.path.isdir(os.path.join(streamData,streamB)):
    print('There exists those folders')
else:
    os.mkdir(streamData)
    os.mkdir(os.path.join(streamData,streamB))
    os.mkdir(os.path.join(streamData,streamA))
    

for folder in os.listdir(parentFolder):
    
    
    
    
    if os.path.isdir(os.path.join(streamData,streamA,folder)) & os.path.isdir(os.path.join(streamData,streamB,folder)):
        pass
    else:
        os.mkdir(os.path.join(streamData,streamA,folder))
        os.mkdir(os.path.join(streamData,streamB,folder))
    files = os.listdir(os.path.join(parentFolder,folder))
    
    
    
    
    for i in range(0,len(files),1):
        source_path = os.path.join(parentFolder,folder,files[i])
        destination_path = os.path.join(streamData,streamA,folder,files[i])
        destination_path2 = os.path.join(streamData,streamB,folder,files[i+1])
        shutil.copy(source_path, destination_path)
        shutil.copy(source_path, destination_path2)
