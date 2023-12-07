import splitfolders

input_folder = 'asl_alphabet_train\\asl_alphabet_train'
splitfolders.ratio(input_folder, output = "asl_alphabet",seed = 0, ratio=(.7,.15,.15))
