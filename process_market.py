import os
from PIL import Image

dataset = './poseCropImgs'
result_dir = os.path.join(dataset, '../../get_keypoints_from')
os.system('rm -rf ' + result_dir)
os.mkdir(result_dir)
for root, dirs, files in os.walk(dataset):
    for file in files:
        if 'jpg' in file:
            full_path = os.path.join(root, file)
            id_name = file.split('_')[0]
            id_dir = os.path.join(result_dir, id_name)
            if not os.path.exists(id_dir):
                os.mkdir(id_dir)
            output_path = os.path.join(id_dir, file)
            flip_output_path = os.path.join(id_dir, 'flip' + file)
            output_file = open(output_path, 'wb')
            im = Image.open(full_path)
            flip_im = im.transpose(Image.FLIP_LEFT_RIGHT)
            flip_im.save(flip_output_path)
            output_file.write(open(full_path, 'rb').read())
