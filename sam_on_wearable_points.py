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
import re

from sam3.model_builder import build_sam3_video_model

mask_clrs = ((0,0,255),(0,255,0),(255,0,0),(0,255,255))
sizes = (6,12,18,24)
def save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path):
    img = inference_state["images"].get_frame(out_frame_idx)
    img = img.permute(1,2,0).cpu().numpy()
    img_min, img_max = img.min(), img.max()
    img = (img-img_min)/(img_max-img_min)
    img = Image.fromarray(np.uint8(img*255)).resize((inference_state["images"].video_width, inference_state["images"].video_height))
    # add output masks
    img = np.array(img)
    ori_img = img.copy()
    for i,oid in enumerate([i for i in video_segments[out_frame_idx] if isinstance(i,int)]):
        mask = np.uint8(video_segments[out_frame_idx][oid].squeeze() > 0.5)
        clrImg = np.zeros(img.shape, img.dtype)
        clrImg[:,:] = mask_clrs[i]
        clrMask = cv2.bitwise_and(clrImg, clrImg, mask=mask)
        # make image with just this object
        img2 = cv2.addWeighted(clrMask, .4, ori_img, .6, 0)
        # make combined image
        img2 = cv2.addWeighted(clrMask, .4, img, .6, 0)
        img = cv2.add(cv2.bitwise_or(img,img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
    extra = ''
    if prompts is not None and out_frame_idx in (p['frame'] for p in prompts.values()):
        prompt = [p for p in prompts.values() if p['frame']==out_frame_idx][0]
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
    Image.fromarray(img).save(pathlib.Path(save_path) / f'frame_{out_frame_idx:05d}_mask{extra}.png')

def propagate(predictor, inference_state, chunk_size, save_path=None, prompts=None, save_range=None):
    # run propagation throughout the video and collect the results in a dict
    prompt_frames = sorted([p['frame'] for p in prompts.values()])
    segments = prompt_frames.copy()
    if segments[0]>0:
        segments = [0]+segments
    if segments[-1]<inference_state["num_frames"]-1:
        segments.append(inference_state["num_frames"])  # NB: on purpose one too high, will be subtracted in the next line
    segments = [[segments[i],segments[i+1]-1] for i in range(0,len(segments)-1)]

    video_segments = {}  # video_segments contains the per-frame segmentation results
    skip_next_prompt = False
    for i,s in enumerate(segments):
        if i==0 and prompt_frames[0]>0:
            reverse = True
            to_prompt = s[1]+1
            skip_next_prompt = True
        else:
            reverse = False
            if skip_next_prompt:
                to_prompt = None
                skip_next_prompt = False
            else:
                to_prompt = s[0]
        if to_prompt is not None:
            video_width = inference_state["video_width"]
            video_height = inference_state["video_height"]
            add_prompt = [p for p in prompts.values() if p['frame']==to_prompt][0]
            for o in add_prompt:
                if o=='frame':
                    continue
                for c,l in zip(add_prompt[o]['coords'],add_prompt[o]['labels']):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=add_prompt['frame'],
                        obj_id=o,
                        points = np.array(c).reshape(-1,2)/[video_width, video_height], # pass as relative coords
                        labels = np.array([l]),  # 1 is positive click, 0 is negative click
                        clear_old_points=False
                    )
        if reverse:
            for out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=s[1], max_frame_num_to_track=s[1]-s[0]+1, reverse=True, propagate_preflight=True):
                video_segments[out_frame_idx] = {out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
                video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
                if save_path and (out_frame_idx in prompt_frames or (save_range and out_frame_idx in save_range)):
                    save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path)
            continue
        for out_frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(inference_state, start_frame_idx=s[0], max_frame_num_to_track=s[1]-s[0]+1, reverse=False, propagate_preflight=True):
            video_segments[out_frame_idx] = {out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}
            video_segments[out_frame_idx]['image_file'] = inference_state['images'].img_paths[min(out_frame_idx,len(inference_state['images'].img_paths)-1)]
            if save_path and (out_frame_idx in prompt_frames or (save_range and out_frame_idx in save_range)):
                save_output_with_prompt(out_frame_idx, prompts, video_segments, save_path)

            if out_frame_idx>0 and out_frame_idx%chunk_size == 0:
                yield video_segments
                video_segments.clear()
    yield video_segments


def extract_last_number_and_fix_fname(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    if match:
        original_num = int(match.group(1))
        corrected_num = original_num - 1    # the frame extractor uses 1-based, whereas we want 0-based
        # Replace the old number with the corrected one in the filename
        new_filename = re.sub(r'_(\d+)\.png$', f'_{corrected_num}.png', filename)
        return corrected_num, new_filename
    return None, filename

def load_prompts_from_folder(folder: pathlib.Path, file_name: str):
    # dict with key (full) filename, containing per object a list of coordinates and associated labels
    prompts: dict[pathlib.Path,list[int,int,tuple[int,int]]] = {}
    prompt_files = list((folder).glob(file_name+"_*_prompts.txt"))
    for pf in prompt_files:
        with open(pf) as f:
            reader = csv.reader(f, delimiter="\t")
            pr = list(reader)
        # get associated frame
        if not (pim:=folder/(pf.stem.removesuffix('_prompts')+'.png')).is_file():
            raise ValueError('missing prompt image file')   # actually it isn't used, but still, strange if it would be missing
        prompt_img,pim = extract_last_number_and_fix_fname(pim.name)

        prompts[pim] = {'frame':None, 0:{'coords':[], 'labels':[]}, 1:{'coords':[], 'labels':[]}, 2:{'coords':[], 'labels':[]}}
        for i,p in enumerate(pr):
            obj_id = 0 if p[0]=='pupil' else 1 if p[0]=='iris' else 2
            label = int(p[3])   # 1 is positive prompt, 0 negative
            point_coord = tuple((int(x) for x in p[1:3]))
            prompts[pim]['frame'] = prompt_img
            prompts[pim][obj_id]['coords'].append(point_coord)
            prompts[pim][obj_id]['labels'].append(label)
    return prompts


if __name__ == '__main__':
    input_dirs   = [pathlib.Path(r"D:\datasets\pupil_validation")]
    prompts_base = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\2025 SAM2_3\highres\prompts\SAM3")
    output_base  = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\2025 SAM2_3\highres\output\SAM3_point_prompts")
    run_reversed = False

    # Path containing the videos (zip files or subdirectory of videos)
    subject_folders = [pathlib.Path(f.path) for d in input_dirs for f in os.scandir(d) if f.is_dir()]
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

        if not (prompts_base / subject.name).exists():
            print(f"No prompts found for subject {subject.name}, skipping.")
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

                prompts = load_prompts_from_folder(prompts_base / subject.name, video_file.name)

                inference_state = tracker.init_state(video_path=str(video_file), lazy_loading=True)

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