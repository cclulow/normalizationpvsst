import ffmpeg
import os
import sys

n_runs = 8

# catch the vids... eventually args to this function
# hardcode stuff for now
base_dir = '/media/maclean/Storage/'
mice = ['42R']
dates = ['072224']
subfolder_to_save = '../'
dropped_frame_folder = 'dropped'
for date in dates:
    for mouse in mice:
        dir_name = f'{base_dir}/{mouse}/{date}/facecam/'
        fnames = os.listdir(dir_name)
        if not os.path.exists(dir_name + subfolder_to_save):
            os.mkdir(dir_name + subfolder_to_save)

        # deal with dropped frames by moving them all to a subfolder
        # to later read which frames they were, insert appropriate nans
        if not os.path.exists(dir_name + dropped_frame_folder):
            os.mkdir(dir_name + dropped_frame_folder)
        os.system(f'find {dir_name} -type f -size -500k -exec mv "{{}}" {dir_name + dropped_frame_folder}/ \;')

        # now do ffmpeg on remaining files, which should complete
        inpts = ffmpeg.input(dir_name + f'/run_*.tif', pattern_type='glob', framerate=20).filter(
            'eq', gamma=2.0
        )
        output = inpts.output(dir_name + f'{subfolder_to_save}/face_vid.avi', vcodec='libx264', crf=0,
                    pix_fmt='yuv420p')
        output.run()

        # no longer process run-by-run, group all together
        '''
        for run in range(1, n_runs+1):
            run_fnames = [fname for fname in fnames if f'run_{run:02}' in fname]
            print(f'{len(run_fnames)} images for {date}, mouse {mouse}, run {run}')
            run_fnames = sorted(run_fnames)

            # ffmpeg work
            #inpts = [ffmpeg.input(dir_name + fname) for fname in run_fnames]
            inpts = ffmpeg.input(dir_name + f'/run_{run:02}*.tif', pattern_type='glob', framerate=20)
            #fr = ffmpeg.filter(inpts, 'setpts', '10')
            output = inpts.output(dir_name + f'{subfolder_to_save}/vid_{run:02}.avi', vcodec='libx264rgb', crf=0,
                    pix_fmt='yuv420p')
            output.run()
        '''

