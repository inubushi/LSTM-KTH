# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 17:27:45 2017

@author: chamin

This program partitions the KTH dataset into a set of folders and extracts
frames from the videos. The objective is to use the frames for activity
recognition.

"""

# system commands
import os

# natural sorting
import re

_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# change here to specify the absolute path to the top folder of your data
trg_data_root = "/path/to/dataset/KTH/"

# a few other constants
class_labels = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"] # 6 labels
frame_path = "/frames/"
video_path = "/videos/"
frame_set_prefix = "person" # 2 digit person ID [01..25] follows
person_count = 25 # 25 persons in the full dataset

rec_prefix = "d" # seq ID [1..4] follows
rec_count = 4
seg_prefix = "seg" # seq ID [1..4] follows
seg_count = 4

# create folders for frame sequences, in advance
for x in range(0, len(class_labels)):
    # floder for each class
    class_folder = trg_data_root + class_labels[x]

    # we will be creating these two subfolders in each class
    class_frame_path = trg_data_root + class_labels[x] + frame_path
    class_frame_path_cmd = "mkdir " + class_frame_path

    # create folders
    os.system(class_frame_path_cmd)

# open the text file containing segment indices
indices_file = open("KTH_sequences.txt", "r")

# read the file and process the videos
for j in range(1, person_count+1):
    # read one line from file; added for readability
    person_name = indices_file.readline()
    print "person ", j

    for i in range(0, len(class_labels)):

        # add indicator
        print class_labels[i], "\t",

        # floder for each class
        class_folder = trg_data_root + class_labels[i]
        class_frame_path = trg_data_root + class_labels[i] + frame_path

        # person prefix for each filename
        if j<10:
            person_prefix = "person0" + str(j) + "_" + class_labels[i] + "_"
        else:
            person_prefix = "person" + str(j) + "_" + class_labels[i] + "_"

        # loop over all recordings for each person
        for k in range(1,rec_count+1):
            # file name of recording
            rec_filename = class_folder + "/" + person_prefix + "d" + str(k) + "_uncomp.avi"

            #subfolder for frames
            person_subfolder = person_prefix + "d" + str(k)

            # output folder for the frames
            output_folder = class_frame_path + person_subfolder
            output_folder_cmd = "mkdir " + output_folder

            # read recording name, for verification
            recording_name = indices_file.readline()

            # read the segment indices line, too
            seg_line = indices_file.readline()

            # check for proper folder names and process
            if (recording_name.rstrip() == person_subfolder and j>20):
                # add indicator
                print k,
                # use ffmpeg to create frames
                ffmpeg_cmd = "ffmpeg -i " + rec_filename + " " + output_folder + "/frame%d.jpg"

                # execute both commands at once - being lazy, I assume no error ;-)
                os.system(output_folder_cmd)
                os.system(ffmpeg_cmd)

                # process indices
                segments = seg_line.split(', ')
                for p in range(0, seg_count):
                    # folder name for segment
                    seg_name = seg_prefix + str(p+1)
                    seg_folder = output_folder + "/" + seg_name
                    seg_folder_cmd = "mkdir " + seg_folder

                    #create the folder
                    os.system(seg_folder_cmd)

                    seg_string = segments[p].rstrip()
                    start_and_finish = seg_string.split('-')

                    # mov frames in a loop
                    for q in range(int(start_and_finish[0]), int(start_and_finish[1])+1):
                        # move the frames
                        frame_name = "frame" + str(q) + ".jpg"
                        source_frame = output_folder + "/" + frame_name
                        move_cmd = "mv " + source_frame + " " + seg_folder
                        os.system(move_cmd)

                # remove unnecessary frames
                remove_frames_cmd = "rm " + output_folder + "/*.jpg"
                os.system(remove_frames_cmd)

            else:
                print rec_filename
                print "invalid index: skipping this recording"

        # done with this person
        print ""

# one video is missing in the original dataset. We use its neighbor&s frameset, as a quick fix
missing_folder = trg_data_root + "handclapping/frames/person13_handclapping_d3"
neighbor_folder = trg_data_root + "handclapping/frames/person13_handclapping_d2"
copy_cmd = "cp -r " + neighbor_folder + " " + missing_folder
os.system(copy_cmd)

# cleaning up
indices_file.close()
