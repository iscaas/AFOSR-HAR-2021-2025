import os
import csv

def save_video_paths_with_directory_numbers(root_directory, csv_filename):
    
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(['Video Path', 'Directory Number'])
        
        directory_map = {}
        directory_number = 1
        
        
        for subdir, _, files in os.walk(root_directory):
            
            if subdir not in directory_map:
                directory_map[subdir] = directory_number
                directory_number += 1
            
            for file in files:
                
                if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                    
                    video_path = os.path.join(subdir, file)
                    
                    dir_number = directory_map[subdir]
                    
                    writer.writerow([video_path, dir_number])
                    
    print(f"Video paths and directory numbers have been saved to {csv_filename}")

data_directory = 'dataset_name/train/'
csv_filename = 'dataset_name/train.csv'

save_video_paths_with_directory_numbers(root_directory, csv_filename)

