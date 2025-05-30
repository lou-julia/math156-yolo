import os
from scipy.io import loadmat


def convert_vbb_to_yolo(vbb_path, output_dir, image_width=640, image_height=480):
   print(f"Reading .vbb file: {vbb_path}")
   vbb = loadmat(vbb_path)
   obj_lists = vbb['A'][0][0][1][0]  # objLists
   n_frames = vbb['A'][0][0][0][0][0]  # nFrame
   obj_labels = vbb['A'][0][0][4][0]  # get objects label
   label_map = [str(x[0]) for x in obj_labels]

   abs_output_dir = os.path.abspath(output_dir)
   os.makedirs(abs_output_dir, exist_ok=True)


   for frame_idx in range(n_frames):
       if obj_lists[frame_idx].size == 0:
           objs = []
       else:
           objs = obj_lists[frame_idx][0]


       label_lines = []


       for obj in objs:
           id_ = obj[0][0][0]
           label = label_map[id_ - 1]

           if label != 'person':  # only keep 'person'
              continue

           occluded = obj[3][0][0]  # ingnore occluded ones
           if occluded:
              continue

           pos = obj[1][0]  # [x, y, w, h]
           x, y, w, h = pos
                       
           if w * h < 10:  # small object filter
                continue


           # Convert to YOLO format (center x/y, normalized)
           x_center = (x + w / 2) / image_width
           y_center = (y + h / 2) / image_height
           w_norm = w / image_width
           h_norm = h / image_height


           label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")


       vbb_name = os.path.splitext(os.path.basename(vbb_path))[0]  # e.g., V000
       txt_path = os.path.join(abs_output_dir, f"{vbb_name}_{frame_idx:05d}.txt")

       with open(txt_path, 'w') as f:
           f.write('\n'.join(label_lines))


   print(f"Done. {n_frames} label files saved in:\n{abs_output_dir}")



NUMBER_OF_SETS = 10

## Okay so here is what you need to know about using this, you really only need to change the root_dir path, vbb_path, and output dir
## I can make this a bit more user friendly when I get the chance but basicaally I assume you already have a file called annotations to put the txt files into, and this location is what goes into output dir
## okay but notice that you dont change the entire dir, it should be like '<location>' + /set{s:02d}/V{v:03d}.vbb for the vbb path and output path, +/set{s:02d} for the root dir. I know I can use os.path.join to make renaming the location more user friendly, which will proably be the next thing I do when I get the chance

for s in range(NUMBER_OF_SETS + 1):
    root_dir = f'C:/Users/dasul/Downloads/data_and_labels/annotations/annotations/set{s:02d}'
    vbb_files = os.listdir(root_dir)
    NUMBER_OF_VB = len(vbb_files)
    print(NUMBER_OF_VB)
    for v in range(NUMBER_OF_VB):
        vbb_path = f'C:/Users/dasul/Downloads/data_and_labels/annotations/annotations/set{s:02d}/V{v:03d}.vbb'
        print(vbb_path)
        output_dir = f'C:/Users/dasul/Downloads/annotations/set{s:02d}/V{v:03d}'
        convert_vbb_to_yolo(vbb_path, output_dir, image_width=640, image_height=480)
