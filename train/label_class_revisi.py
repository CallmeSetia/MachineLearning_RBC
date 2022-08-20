import glob
import os
import re
   
txt_file_paths = glob.glob(r"data/obj/*.txt")
for i, file_path in enumerate(txt_file_paths):
    # get image size
    with open(file_path, "r") as f_o:
        lines = f_o.readlines()

        text_converted = []
        for line in lines:
            # print(line)
            numbers = re.findall("[0-9.]+", line)
            # print(numbers)

            if numbers:
              if numbers[0] == '15':
                numbers[0] = 0
              if numbers[0] == '16':
                numbers[0] = 1
              if numbers[0] == '17':
                numbers[0] = 2
              
              # Define coordinates
              text = "{} {} {} {} {}".format(numbers[0], numbers[1], numbers[2], numbers[3], numbers[4])
              text_converted.append(text)
              # print(i, file_path)
              print(text)
        # Write file
        with open(file_path, 'w') as fp:
            for item in text_converted:
                fp.writelines("%s\n" % item)
