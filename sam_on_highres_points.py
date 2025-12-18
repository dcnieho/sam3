import pathlib
import cv2
from PIL import Image
import numpy as np
import pickle
import compress_pickle
import pathlib
import traceback
import gc
import natsort
import csv
import os
import torch

from sam3.model_builder import build_sam3_video_model

mask_clrs = ((0,0,255),(0,255,0),(255,0,0),(0,255,255))
sizes = (6,12,18,24)
def save_output_with_prompt(actual_fr_idx, out_frame_idx, prompts, video_segments, save_path):
    img = inference_state["images"].get_frame(out_frame_idx)
    img = img.permute(1,2,0).cpu().numpy()
    img_min, img_max = img.min(), img.max()
    img = (img-img_min)/(img_max-img_min)
    img = Image.fromarray(np.uint8(img*255)).resize((inference_state["images"].video_width, inference_state["images"].video_height))
    # add output masks
    img = np.array(img)
    ori_img = img.copy()
    for i,oid in enumerate([i for i in video_segments[actual_fr_idx] if isinstance(i,int)]):
        mask = np.uint8(video_segments[actual_fr_idx][oid].squeeze() > 0.5)
        clrImg = np.zeros(img.shape, img.dtype)
        clrImg[:,:] = mask_clrs[i]
        clrMask = cv2.bitwise_and(clrImg, clrImg, mask=mask)
        # make image with just this object
        img2 = cv2.addWeighted(clrMask, .4, ori_img, .6, 0)
        # make combined image
        img2 = cv2.addWeighted(clrMask, .4, img, .6, 0)
        img = cv2.add(cv2.bitwise_or(img,img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
    extra = ''
    if prompts is not None and actual_fr_idx<0:
        prompt = prompts[video_segments[actual_fr_idx]['image_file']]
        extra = '_prompt'
        for o in prompt:
            if o=='frame':
                continue
            pr = prompt[o]
            clr = mask_clrs[o]
            for c,l in zip(pr['coords'],pr['labels']):
                p = [int(x) for x in c]
                if l==1:
                    img = cv2.drawMarker(img, (p[0], p[1]), clr, cv2.MARKER_CROSS, 6, 2)
                else:
                    img = cv2.drawMarker(img, (p[0], p[1]), clr, cv2.MARKER_SQUARE, sizes[o], 2)
    Image.fromarray(img).save(pathlib.Path(save_path) / f'frame_{actual_fr_idx:05d}_mask{extra}.png')

def propagate(predictor, inference_state, chunk_size, save_path=None, prompts=None, save_range=None):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=999999999, reverse=False, propagate_preflight=True):
        actual_fr_idx = out_frame_idx-(len(prompts) if prompts else 0)
        video_segments[actual_fr_idx] = {out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
        video_segments[actual_fr_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
        if save_path and (actual_fr_idx<0 or (save_range and actual_fr_idx in save_range)):
            save_output_with_prompt(actual_fr_idx, out_frame_idx, prompts, video_segments, save_path)

        if actual_fr_idx>0 and actual_fr_idx%chunk_size == 0:
            yield video_segments
            video_segments.clear()
    yield video_segments


def load_prompts_from_folder(folder: pathlib.Path):
    # prompts are stored in text files
    prompt_files = list(folder.glob("*_prompts.txt"))
    prompt_files = natsort.natsorted(prompt_files)
    # dict with key (full) filename, containing per object a list of coordinates and associated labels
    prompts: dict[pathlib.Path,list[int,int,tuple[int,int]]] = {}
    for fp in prompt_files:
        with open(fp) as f:
            reader = csv.reader(f, delimiter="\t")
            pr = list(reader)
        file = fp.with_name('_'.join(fp.stem.split('_')[:4])+'.png')
        prompts[file] = {0:{'coords':[], 'labels':[]}, 1:{'coords':[], 'labels':[]}, 2:{'coords':[], 'labels':[]}, 3:{'coords':[], 'labels':[]}}
        for p in pr:
            obj_id = 0 if p[0]=='CR' else 1 if p[0]=='pupil' else 2 if p[0]=='iris' else 3
            label = int(p[3])   # 1 is positive prompt, 0 negative
            point_coord = tuple((int(x) for x in p[1:3]))
            prompts[file][obj_id]['coords'].append(point_coord)
            prompts[file][obj_id]['labels'].append(label)
    return prompts


if __name__ == '__main__':
    input_dir   = pathlib.Path(r"D:\datasets\sean datasets\2023-04-25_1000Hz_100_EL")
    prompts_base = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\2025 SAM2_3\highres\prompts\SAM3")
    output_base  = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\2025 SAM2_3\highres\output\SAM3_point_prompts")
    run_reversed = False

    # Path containing the videos (zip files or subdirectory of videos)
    subject_folders = [pathlib.Path(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    subject_folders = natsort.natsorted(subject_folders, reverse=run_reversed)

    sam3_model = build_sam3_video_model(checkpoint_path=pathlib.Path(r'C:\Users\Dee\Desktop\sam3\checkpoints\sam3.pt'))
    tracker = sam3_model.tracker
    tracker.backbone = sam3_model.detector.backbone
    chunk_size = 10000  # store to file once this many frames are processed
    for subject in subject_folders:
        print(f"############## {subject.name} ##############")
        video_files = list(subject.glob("*.mp4"))
        video_files = natsort.natsorted(video_files, reverse=run_reversed)
        if not video_files:
            print(f"No video files found for subject {subject.name}, skipping.")
            continue
        for i,video_file in enumerate(video_files):
            try:
                this_output_path = output_base / subject.name / video_file.stem
                print(f"############## {this_output_path} ##############")
                this_output_path.mkdir(parents=True, exist_ok=True)

                savepath_videosegs = this_output_path / 'segments_0.pickle.gz'
                if os.path.exists(savepath_videosegs):
                    print(f"Already done. Skipping {subject.name}/{video_file.name}")
                    continue

                prompts = load_prompts_from_folder(prompts_base / subject.name)

                inference_state = tracker.init_state(video_path=str(video_file), lazy_loading=True, separate_prompts=prompts)

                # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
                to_save = {*range(0,1200,10), *range(1200,1000000,100)}
                for i,video_segments in enumerate(propagate(tracker, inference_state, chunk_size, this_output_path, prompts, to_save)):
                    savepath_videosegs = this_output_path / f'segments_{i}.pickle.gz'
                    with open(savepath_videosegs, 'wb') as handle:
                        compress_pickle.dump(video_segments, handle, pickler_kwargs={'protocol': pickle.HIGHEST_PROTOCOL})
                    video_segments.clear()

                del inference_state
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                if 'inference_state' in globals():
                    del inference_state
                gc.collect()
                torch.cuda.empty_cache()

                error_message = f'Failed: {video_file} due to error.'
                print(error_message)
                print(f"An error occurred: {e}")
                print("Error type: %s", type(e).__name__)
                print("Detailed traceback:\n%s", traceback.format_exc())