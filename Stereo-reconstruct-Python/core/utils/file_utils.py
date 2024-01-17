
import os
import time
import shutil
import numpy as np
import json
import glob
import random
import subprocess
import concurrent.futures
import numbers
from datetime import datetime


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', 'y', '1')


def get_time(format="S"):
    """
    :param format:
    :return:
    """
    if format in ["S", "s"]:
        # time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        time = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
    elif format in ["P", "p"]:

        time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S_%f')
        time = time[:-2]
    else:
        time = (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
    return time


def get_kwargs_name(**kwargs):
    prefix = []
    for k, v in kwargs.items():
        if isinstance(v, list):
            v = [str(l) for l in v]
            prefix += v
        else:
            f = "{}_{}".format(k, v)
            prefix.append(f)
    prefix = "_".join(prefix)
    return prefix


def combine_flags(flags: list, use_time=True, info=True):
    """
    :param flags:
    :param info:
    :return:
    """
    out_flags = []
    for f in flags:
        if isinstance(f, dict):
            f = get_kwargs_name(**f)
        out_flags.append(f)
    if use_time:
        out_flags += [get_time()]
    out_flags = [str(f) for f in out_flags if f]
    out_flags = "_".join(out_flags)
    if info:
        print(out_flags)
    return out_flags


class WriterTXT(object):
    """ write data in txt files"""

    def __init__(self, filename, mode='w'):
        self.f = None
        if filename:
            self.f = open(filename, mode=mode)

    def write_line_str(self, line_str, endline="\n"):
        if self.f:
            line_str = line_str + endline
            self.f.write(line_str)
            self.f.flush()

    def write_line_list(self, line_list, endline="\n"):
        if self.f:
            for line_list in line_list:

                line_str = " ".join('%s' % id for id in line_list)
                self.write_line_str(line_str, endline=endline)
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def parser_classes(class_name):
    """
    :return:
    """
    if isinstance(class_name, str):
        class_name = read_data(class_name, split=None)
    elif isinstance(class_name, numbers.Number):
        class_name = [str(i) for i in range(int(class_name))]
    if isinstance(class_name, list):
        class_dict = {str(class_name): i for i, class_name in enumerate(class_name)}
    elif isinstance(class_name, dict):
        class_dict = class_name
    else:
        class_dict = None
    return class_name, class_dict


def read_json_data(json_path):

    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def write_json_path(out_json_path, json_data):

    with open(out_json_path, 'w', encoding="utf-8") as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)


def write_data(filename, content_list, split=" ", mode='w'):

    with open(filename, mode=mode, encoding='utf-8') as f:
        for line_list in content_list:
            # 将list转为string
            line = "{}".format(split).join('%s' % id for id in line_list)
            f.write(line + "\n")
        f.flush()


def write_list_data(filename, list_data, mode='w'):

    with open(filename, mode=mode, encoding='utf-8') as f:
        for line in list_data:
            # 将list转为string
            f.write(str(line) + "\n")
        f.flush()


def read_data(filename, split=" ", convertNum=True):

    with open(filename, mode="r", encoding='utf-8') as f:
        content_list = f.readlines()
        if split is None:
            content_list = [content.rstrip() for content in content_list]
            return content_list
        else:
            content_list = [content.rstrip().split(split) for content in content_list]
        if convertNum:
            for i, line in enumerate(content_list):
                line_data = []
                for l in line:
                    if is_int(l):  
                        line_data.append(int(l))
                    elif is_float(l):  
                        line_data.append(float(l))
                    else:
                        line_data.append(l)
                content_list[i] = line_data
    return content_list


def read_line_image_label(line_image_label):
    '''
    line_image_label:[image_id,boxes_nums,x1, y1, w, h, label_id,x1, y1, w, h, label_id,...]
    :param line_image_label:
    :return:
    '''
    line_image_label = line_image_label.strip().split()
    image_id = line_image_label[0]
    boxes_nums = int(line_image_label[1])
    box = []
    label = []
    for i in range(boxes_nums):
        x = float(line_image_label[2 + 5 * i])
        y = float(line_image_label[3 + 5 * i])
        w = float(line_image_label[4 + 5 * i])
        h = float(line_image_label[5 + 5 * i])
        c = int(line_image_label[6 + 5 * i])
        if w <= 0 or h <= 0:
            continue
        box.append([x, y, x + w, y + h])
        label.append(c)
    return image_id, box, label


def read_lines_image_labels(filename):
    """
    :param filename:
    :return:
    """
    boxes_label_lists = []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            image_id, box, label = read_line_image_label(line)
            boxes_label_lists.append([image_id, box, label])
    return boxes_label_lists


def is_int(str):

    try:
        x = int(str)
        return isinstance(x, int)
    except ValueError:
        return False


def is_float(str):

    try:
        x = float(str)
        return isinstance(x, float)
    except ValueError:
        return False


def list2str(content_list):
    """
    convert list to string
    :param content_list:
    :return:
    """
    content_str_list = []
    for line_list in content_list:
        line_str = " ".join('%s' % id for id in line_list)
        content_str_list.append(line_str)
    return content_str_list


def get_basename(file_list):
    """
    get files basename
    :param file_list:
    :return:
    """
    dest_list = []
    for file_path in file_list:
        basename = os.path.basename(file_path)
        dest_list.append(basename)
    return dest_list


def randam_select_images(image_list, nums, shuffle=True):
    """
    randam select nums images
    :param image_list:
    :param nums:
    :param shuffle:
    :return:
    """
    image_nums = len(image_list)
    if image_nums <= nums:
        return image_list
    if shuffle:
        random.seed(100)
        random.shuffle(image_list)
    out = image_list[:nums]
    return out


def remove_dir(dir):
    """
    remove directory
    :param dir:
    :return:
    """
    if os.path.exists(dir):
        shutil.rmtree(dir)


def get_prefix_files(file_dir, prefix):
    """
    :param file_dir:
    :param prefix: "best*"
    :return:
    """
    file_list = glob.glob(os.path.join(file_dir, prefix))
    return file_list


def remove_prefix_files(file_dir, prefix):
    """
    :param file_dir:
    :param prefix: "best*"
    :return:
    """
    file_list = get_prefix_files(file_dir, prefix)
    for file in file_list:
        if os.path.isfile(file):
            remove_file(file)
        elif os.path.isdir(file):
            remove_dir(file)


def remove_file(path):
    """
    remove files
    :param path:
    :return:
    """
    if os.path.exists(path):
        os.remove(path)


def remove_file_list(file_list):
    """
    remove file list
    :param file_list:
    :return:
    """
    for file_path in file_list:
        remove_file(file_path)


def copy_dir_multi_thread(sync_source_root, sync_dest_dir, dataset, max_workers=1):
    """
    :param sync_source_dir:
    :param sync_dest_dir:
    :param dataset:
    :return:
    """

    def rsync_cmd(source_dir, dest_dir):
        cmd_line = "rsync -a {0} {1}".format(source_dir, dest_dir)
        # subprocess.call(cmd_line.split())
        subprocess.call(cmd_line)

    sync_dest_dir = sync_dest_dir.rstrip('/')

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_rsync = {}
        for source_dir in dataset:
            sync_source_dir = os.path.join(sync_source_root, source_dir.strip('/'))
            future_to_rsync[executor.submit(rsync_cmd, sync_source_dir, sync_dest_dir)] = source_dir

        for future in concurrent.futures.as_completed(future_to_rsync):
            source_dir = future_to_rsync[future]
            try:
                _ = future.result()
            except Exception as exc:
                print("%s copy data generated an exception: %s" % (source_dir, exc))
            else:
                print("%s copy data successful." % (source_dir,))


def copy_dir_delete(src, dst):
    """
    copy src directory to dst directory,will detete the dst same directory
    :param src:
    :param dst:
    :return:
    """
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # time.sleep(3 / 1000.)


def copy_dir(src, dst):
    """ copy src-directory to dst-directory, will cover the same files"""
    if not os.path.exists(src):
        print("\nno src path:{}".format(src))
        return
    for root, dirs, files in os.walk(src, topdown=False):
        dest_path = os.path.join(dst, os.path.relpath(root, src))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        for filename in files:
            copy_file(
                os.path.join(root, filename),
                os.path.join(dest_path, filename)
            )


def move_file(srcfile, dstfile):

    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile) 
        if not os.path.exists(fpath):
            os.makedirs(fpath)  
        shutil.move(srcfile, dstfile)
        # print("copy %s -> %s"%( srcfile,dstfile))
        # time.sleep(1 / 1000.)


def copy_file(srcfile, dstfile):
    """
    copy src file to dst file
    :param srcfile:
    :param dstfile:
    :return:
    """
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile, dstfile)  
        # print("copy %s -> %s"%( srcfile,dstfile))
        # time.sleep(1 / 1000.)


def copy_file_to_dir(srcfile, des_dir):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  
        if not os.path.exists(des_dir):
            os.makedirs(des_dir) 
        dstfile = os.path.join(des_dir, fname)
        shutil.copyfile(srcfile, dstfile) 


def move_file_to_dir(srcfile, des_dir):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile) 
        if not os.path.exists(des_dir):
            os.makedirs(des_dir) 
        dstfile = os.path.join(des_dir, fname)

        move_file(srcfile, dstfile) 


def merge_dir(src, dst, sub, merge_same):
    src_dir = os.path.join(src, sub)
    dst_dir = os.path.join(dst, sub)

    if not os.path.exists(src_dir):
        print("\nno src path:{}".format(src))
        return
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    elif not merge_same:
        t = get_time()
        dst_dir = os.path.join(dst, sub + "_{}".format(t))
        print("have save sub:{}".format(dst_dir))
    copy_dir(src_dir, dst_dir)


def create_dir(parent_dir, dir1=None, filename=None):
    """
    create directory
    :param parent_dir:
    :param dir1:
    :param filename:
    :return:
    """
    out_path = parent_dir
    if dir1:
        out_path = os.path.join(parent_dir, dir1)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if filename:
        out_path = os.path.join(out_path, filename)
    return out_path


def create_file_path(filename):
    """
    create file in path
    :param filename:
    :return:
    """
    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)
    out_path = create_dir(dirname, dir1=None, filename=basename)
    return out_path


def get_sub_directory_list(input_dir):

    sub_list = []
    for root, dirs, files in os.walk(input_dir):
        sub_list = dirs
        break

    sub_list.sort()
    return sub_list


def get_files_lists(image_dir, subname="", postfix=["*.jpg", "*.png"], shuffle=False):

    if isinstance(image_dir, list):
        image_list = image_dir
    elif image_dir.endswith(".txt"):
        data_root = os.path.dirname(image_dir)
        image_list = read_data(image_dir)
        if subname:
            image_list = [os.path.join(data_root, subname, str(n[0]) + postfix[0][1:]) for n in image_list]
    elif os.path.isdir(image_dir):
        image_list = get_files_list(image_dir, prefix="", postfix=postfix)
    elif os.path.isfile(image_dir):
        image_list = [image_dir]
    else:
        raise Exception("Error:{}".format(image_dir))
    if shuffle:
        random.seed(100)
        random.shuffle(image_list)
    return image_list


def get_files_list(file_dir, prefix="", postfix=None, basename=False):

    file_list = []
    if not postfix:
        file_list = glob.glob(os.path.join(file_dir, prefix + "*"))
    else:
        postfix = [postfix] if isinstance(postfix, str) else postfix
        for p in postfix:
            dir = os.path.join(file_dir, prefix + p)
            item = glob.glob(dir)
            file_list = file_list + item if item else file_list
    file_list.sort()
    file_list = get_basename(file_list) if basename else file_list
    return file_list


def get_images_list(file_dir, prefix="", postfix=["*.png", "*.jpg"], basename=False):

    return get_files_list(file_dir, prefix=prefix, postfix=postfix, basename=basename)


def get_files_labels(file_dir, prefix="", postfix=["*.png", "*.jpg"], basename=False):

    file_list = get_files_list(file_dir, prefix=prefix, postfix=postfix, basename=basename)
    label_list = []
    for filePath in file_list:
        label = filePath.split(os.sep)[-2]
        label_list.append(label)
    return file_list, label_list


def decode_label(label_list, name_table):

    name_list = []
    for label in label_list:
        name = name_table[label]
        name_list.append(name)
    return name_list


def encode_label(name_list, name_table, unknow=0):

    label_list = []
    # name_table = {name_table[i]: i for i in range(len(name_table))}
    for name in name_list:
        if name in name_table:
            index = name_table.index(name)
        else:
            index = unknow
        label_list.append(index)
    return label_list


def list2dict(data):
    """
    convert list to dict
    :param data:
    :return:
    """
    data = {data[i]: i for i in range(len(data))}
    return data


def print_dict(dict_data, save_path):
    """
    print dict info
    :param dict_data:
    :param save_path:
    :return:
    """
    list_config = []
    for key in dict_data:
        info = "conf.{}={}".format(key, dict_data[key])
        print(info)
        list_config.append(info)
    if save_path is not None:
        with open(save_path, "w") as f:
            for info in list_config:
                f.writelines(info + "\n")


def read_pair_data(filename, split=True):
    '''
    read pair data,data:[image1.jpg image2.jpg 0]
    :param filename:
    :param split:
    :return:
    '''
    content_list = read_data(filename)
    if split:
        content_list = np.asarray(content_list)
        faces_list1 = content_list[:, :1].reshape(-1)
        faces_list2 = content_list[:, 1:2].reshape(-1)
        # convert to 0/1
        issames_data = np.asarray(content_list[:, 2:3].reshape(-1), dtype=np.int)
        issames_data = np.where(issames_data > 0, 1, 0)
        faces_list1 = faces_list1.tolist()
        faces_list2 = faces_list2.tolist()
        issames_data = issames_data.tolist()
        return faces_list1, faces_list2, issames_data
    return content_list


def check_files(files_list, sizeTh=1 * 1024, isRemove=False):

    i = 0
    while i < len(files_list):
        path = files_list[i]

        if not (os.path.exists(path)):
            print(" non-existent file:{}".format(path))
            files_list.pop(i)
            continue

        f_size = os.path.getsize(path)
        if f_size < sizeTh:
            print(" empty file:{}".format(path))
            if isRemove:
                os.remove(path)
                print(" info:----------------remove image_dict:{}".format(path))
            files_list.pop(i)
            continue
        i += 1
    return files_list


def get_set_inter_union_diff(set1, set2):


    difference = list(set(set1) ^ set(set2))

    intersection = list(set(set1) & set(set2))

    union = list(set(set1) | set(set2))
    dset1 = list(set(set1) - set(set2))
    dset2 = list(set(set2) - set(set1))
    return intersection, union, difference


def get_files_id(file_list):
    """
    :param file_list:
    :return:
    """
    image_idx = []
    for path in file_list:
        basename = os.path.basename(path)
        id = basename.split(".")[0]
        image_idx.append(id)
    return image_idx


def get_loacl_eth2():

    eth_list = []
    os.system("ls -l /sys/class/net/ | grep -v virtual | sed '1d' | awk 'BEGIN {FS=\"/\"} {print $NF}' > eth.yaml")
    try:
        with open('./eth.yaml', "r") as f:
            for line in f.readlines():
                line = line.strip()
                eth_list.append(line.lower())
    except Exception as e:
        print(e)
        eth_list = []
    return eth_list


def get_loacl_eth():

    eth_list = []
    cmd = "ls -l /sys/class/net/ | grep -v virtual | sed '1d' | awk 'BEGIN {FS=\"/\"} {print $NF}'"
    try:
        with os.popen(cmd) as f:
            for line in f.readlines():
                line = line.strip()
                eth_list.append(line.lower())
    except Exception as e:
        print(e, "can not found eth,will set default eth is:eth0")
        eth_list = ["eth0"]
    if not eth_list:
        eth_list = ["eth0"]
    return eth_list


def merge_files(files_list):

    content_list = []
    for file in files_list:
        data = read_data(file)

    return content_list


def multi_thread_task(content_list, func, num_processes=4, remove_bad=False, Async=True, **kwargs):

    from multiprocessing.pool import ThreadPool

    pool = ThreadPool(processes=num_processes)
    thread_list = []
    for item in content_list:
        if Async:
            out = pool.apply_async(func=func, args=(item,), kwds=kwargs) 
        else:
            out = pool.apply(func=func, args=(item,), kwds=kwargs)  
        thread_list.append(out)

    pool.close()
    pool.join()

    dst_content_list = []
    if Async:
        for p in thread_list:
            image = p.get() 
            dst_content_list.append(image)
    else:
        dst_content_list = thread_list
    if remove_bad:
        dst_content_list = [i for i in dst_content_list if i is not None]
    return dst_content_list


if __name__ == '__main__':
    parent = "/home/dm/data3/Project/3D/Camera-Calibration-Reconstruct/data/temp"
    dir_list = get_files_list(parent)
    print(dir_list)
