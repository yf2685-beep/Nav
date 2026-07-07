#!/usr/bin/env python3
# Python 3 port of the official Matterport3D download script (download_mp.py).
# Only mechanical changes: print(), urllib.request, input(), bytes.decode(), and an
# --agree_tos flag so it can run non-interactively (you confirm the MP Terms of Use by
# passing that flag). Behaviour is otherwise identical to the original.
import argparse
import os
import tempfile
import urllib.request

BASE_URL = 'http://kaldir.vc.cit.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    'cameras', 'matterport_camera_intrinsics', 'matterport_camera_poses',
    'matterport_color_images', 'matterport_depth_images', 'matterport_hdr_images',
    'matterport_mesh', 'matterport_skybox_images', 'undistorted_camera_parameters',
    'undistorted_color_images', 'undistorted_depth_images', 'undistorted_normal_images',
    'house_segmentations', 'region_segmentations', 'image_overlap_data',
    'poisson_meshes', 'sens',
]
TASK_FILES = {
    'habitat': ['mp3d_habitat.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'igibson': ['mp3d_for_igibson.zip'],
    'minos': ['mp3d_minos.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
}


def get_release_scans(release_file):
    scans = []
    for scan_line in urllib.request.urlopen(release_file):
        scans.append(scan_line.decode('utf-8').rstrip('\n'))
    return scans


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        print('\t' + url + ' > ' + out_file)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        os.close(fh)
        urllib.request.urlretrieve(url, out_file_tmp)
        os.rename(out_file_tmp, out_file)
    else:
        print('WARNING: skipping download of existing file ' + out_file)


def download_scan(scan_id, out_dir, file_types):
    print('Downloading MP scan ' + scan_id + ' ...')
    os.makedirs(out_dir, exist_ok=True)
    for ft in file_types:
        download_file(BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip',
                      out_dir + '/' + ft + '.zip')
    print('Downloaded scan ' + scan_id)


def download_task_data(task_data, out_dir):
    print('Downloading MP task data for ' + str(task_data) + ' ...')
    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            for filepart in TASK_FILES[task_data_id]:
                localpath = os.path.join(out_dir, filepart)
                os.makedirs(os.path.dirname(localpath), exist_ok=True)
                download_file(BASE_URL + RELEASE_TASKS + '/' + filepart, localpath)
                print('Downloaded task data ' + task_data_id)


def main():
    parser = argparse.ArgumentParser(description='Downloads MP public data release (py3 port).')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--task_data', default=[], nargs='+',
                        help='task data files. Any of: ' + ','.join(TASK_FILES.keys()))
    parser.add_argument('--id', default='ALL', help='scan id to download or ALL')
    parser.add_argument('--type', nargs='+', help='file types. Any of: ' + ','.join(FILETYPES))
    parser.add_argument('--agree_tos', action='store_true',
                        help='confirm you agree to the MP Terms of Use (' + TOS_URL + ') and skip prompts')
    args = parser.parse_args()

    if not args.agree_tos:
        print('By continuing you confirm you have agreed to the MP terms of use: ' + TOS_URL)
        input('Press ENTER to continue, or CTRL-C to exit.')

    release_scans = get_release_scans(BASE_URL + RELEASE + '.txt')
    file_types = args.type if args.type else FILETYPES

    if args.task_data:
        download_task_data(args.task_data, os.path.join(args.out_dir, RELEASE_TASKS))
        print('Done downloading task_data for ' + str(args.task_data))

    if args.id and args.id != 'ALL':
        if args.id not in release_scans:
            print('ERROR: Invalid scan id: ' + args.id)
        else:
            download_scan(args.id, os.path.join(args.out_dir, RELEASE, args.id), file_types)


if __name__ == '__main__':
    main()
