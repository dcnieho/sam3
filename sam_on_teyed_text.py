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

from sam3.model_builder import build_sam3_video_predictor

mask_clrs = ((0,0,255),(0,255,0),(255,0,0),(0,255,255))
sizes = (6,12,18,24)
def save_output_with_prompt(out_frame_idx, predictor, session_id, video_segments, save_path):
    frame_source = predictor._get_session(session_id)["state"]['input_batch'].img_batch
    img = frame_source.get_frame(out_frame_idx)
    img = img.permute(1,2,0).cpu().numpy()
    img_min, img_max = img.min(), img.max()
    img = (img-img_min)/(img_max-img_min)
    img = np.array(Image.fromarray(np.uint8(img*255)).resize((frame_source.video_width, frame_source.video_height)))
    # add output masks
    ori_img = img.copy()
    for i,oid in enumerate([i for i in video_segments[out_frame_idx] if isinstance(i,np.int64)]):
        mask = np.uint8(video_segments[out_frame_idx][oid])
        clrImg = np.zeros(img.shape, img.dtype)
        clrImg[:,:] = mask_clrs[i]
        clrMask = cv2.bitwise_and(clrImg, clrImg, mask=mask)
        # make image with just this object
        img2 = cv2.addWeighted(clrMask, .4, ori_img, .6, 0)
        # make combined image
        img2 = cv2.addWeighted(clrMask, .4, img, .6, 0)
        img = cv2.add(cv2.bitwise_or(img,img,mask=cv2.bitwise_not(mask*255)), cv2.bitwise_or(img2,img2,mask=mask))
    Image.fromarray(img).save(pathlib.Path(save_path) / f'frame_{out_frame_idx:05d}_mask.png')

def propagate(predictor, session_id, prompt_frame, chunk_size, save_path=None, save_range=None):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    if prompt_frame>0:
        # first propagate backward from the prompted frame to the beginning of the video
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction="backward",
                start_frame_index=prompt_frame  # NB: not prompt_frame-1, like this the first output is the frame before the prompt
            )
        ):
            out_frame_idx = response["frame_index"]
            video_segments[out_frame_idx] = {i:m for i,m in zip(response["outputs"]["out_obj_ids"], response["outputs"]["out_binary_masks"])}
            
            if save_path and save_range and out_frame_idx in save_range:
                save_output_with_prompt(out_frame_idx, predictor, session_id, video_segments, save_path)

    # then propagate forward to the end
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=prompt_frame
        )
    ):
        out_frame_idx = response["frame_index"]
        video_segments[out_frame_idx] = {i:m for i,m in zip(response["outputs"]["out_obj_ids"], response["outputs"]["out_binary_masks"])}
        
        if save_path and save_range and out_frame_idx in save_range:
            save_output_with_prompt(out_frame_idx, predictor, session_id, video_segments, save_path)

        if out_frame_idx>0 and out_frame_idx%chunk_size == 0:
            yield video_segments
            video_segments.clear()
    yield video_segments
    

if __name__ == '__main__':
    vid_dir = 'VIDEOS'
    gt_dir = 'ANNOTATIONS'
    input_dir    = pathlib.Path(r"D:\TEyeD")
    prompts_base = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\datasets real\TEyeD\prompts")
    output_base  = pathlib.Path(r"\\et-nas.humlab.lu.se\FLEX\2025 SAM2_3\TEyeD\output\SAM3_text_prompts")
    run_reversed = False

    # Path containing the videos (zip files or subdirectory of videos)
    datasets = [fp for f in os.scandir(input_dir) if (fp:=pathlib.Path(f.path)).is_dir() and all((fp/s).is_dir() for s in [vid_dir,gt_dir])]
    datasets = natsort.natsorted(datasets, reverse=run_reversed)

    predictor = build_sam3_video_predictor(checkpoint_path=pathlib.Path(r'C:\Users\Dee\Desktop\sam3\checkpoints\sam3.pt'))
    predictor.lazy_loading = True
    predictor.lazy_loader = 'deprecated'
    chunk_size = 10000  # store to file once this many frames are processed
    for dataset in datasets:
        # if dataset.name!="pupilValidation_gb_neon_illum":
        #     continue
        print(f"############## {dataset.name} ##############")
        video_files = list((dataset/vid_dir).glob("*.mp4"))
        video_files = natsort.natsorted(video_files, reverse=run_reversed)
        if not video_files:
            print(f"No video files found for subject {dataset.name}, skipping.")

        for i,video_file in enumerate(video_files):
            # if video_file.stem!="cam1_R002":
            #     continue
            try:
                this_output_path = output_base / dataset.name / video_file.stem
                print(f"############## {this_output_path} ##############")
                this_output_path.mkdir(parents=True, exist_ok=True)

                savepath_videosegs = this_output_path / 'segments_0.pickle.gz'
                if os.path.exists(savepath_videosegs):
                    print(f"Already done. Skipping {dataset.name}/{video_file.name}")
                    continue

                response = predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=str(video_file),
                    )
                )
                session_id = response["session_id"]

                # check what frame to prompt
                prompt_frame = 0
                from sam_on_teyed_points import load_prompts_from_folder
                prompts = load_prompts_from_folder(prompts_base / dataset.name, video_file.stem)
                if prompts:
                    prompt_frame = list(prompts.values())[0]['frame']
                else:
                    print(f"No prompts found for {dataset.name}/{video_file.name}, skipping.")
                    continue
                
                resp = predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=prompt_frame,
                        text="pupil",
                    )
                )

                # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
                to_save = {*range(0,1200,10), *range(1200,1000000,100)}
                for i,video_segments in enumerate(propagate(predictor, session_id, prompt_frame, chunk_size, this_output_path, to_save)):
                    savepath_videosegs = this_output_path / f'segments_{i}.pickle.gz'
                    with open(savepath_videosegs, 'wb') as handle:
                        compress_pickle.dump(video_segments, handle, pickler_kwargs={'protocol': pickle.HIGHEST_PROTOCOL})
                    video_segments.clear()

                predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=session_id,
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                if 'session_id' in globals():
                    predictor.handle_request(
                        request=dict(
                            type="close_session",
                            session_id=session_id,
                        )
                    )
                gc.collect()
                torch.cuda.empty_cache()

                error_message = f'Failed: {video_file} due to error.'
                print(error_message)
                print(f"An error occurred: {e}")
                print("Error type: %s", type(e).__name__)
                print("Detailed traceback:\n%s", traceback.format_exc())