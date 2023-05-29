from pathlib import Path

# from multiprocessing import shared_memory
import torch.distributed as torch_dist
import numpy as np
import pickle


def get_file_name_list(path: str) -> list:
    with open(path, mode="r") as rf:
        name_list = rf.read().splitlines()
    return name_list


def get_shared_filelist(filelist_path: str = None, blacklist_path: str = None) -> str:
    ignore_files = None
    if blacklist_path:
        ignore_files = get_file_name_list(blacklist_path)
    if filelist_path:
        filelist = get_file_name_list(filelist_path)
    num_ignored = 0
    ignore_file_set = (
        set([f.split("/")[-1] for f in ignore_files]) if ignore_files else None
    )
    print(f"using {filelist_path=}")
    if ignore_file_set:
        files = [None for _ in range(len(filelist))]
        idx = 0
        for file_name in filelist:
            tmp_path = Path(file_name)
            if tmp_path.name in ignore_file_set:
                num_ignored += 1
                continue
            files[idx] = file_name
            idx += 1
        files = files[:idx]
    else:
        files = filelist
    assert len(files) > 0, "no sound files found"
    print(f"skipped {num_ignored} ignored files")

    # create numpy list
    np_buffer_list = [np.frombuffer(pickle.dumps(x), dtype=np.uint8) for x in files]
    np_address_list = np.cumsum([len(x) for x in np_buffer_list])
    np_list = np.concatenate(np_buffer_list)
    # clean up
    del files, ignore_files, ignore_file_set, filelist, np_buffer_list
    return np_list, np_address_list


def get_distributed_shared_filelist(
    world_size: int,
    rank: int,
    local_rank: int,
    filelist_path: str,
    blacklist_path: str,
) -> str:
    """get the name of shared memory"""
    shared_obj_list = [None, None]
    # get data
    if rank == 0:
        tmp_np_list, tmp_np_addr_list = get_shared_filelist(
            filelist_path=filelist_path,
            blacklist_path=blacklist_path,
        )
        shared_obj_list[0] = tmp_np_list
        shared_obj_list[1] = tmp_np_addr_list
    print(f"before: {rank=} {local_rank=} {shared_obj_list=}")
    # gather
    if torch_dist.is_initialized():
        torch_dist.broadcast_object_list(shared_obj_list, src=0)
    print(f"after: {rank=} {local_rank=} {shared_obj_list=}")
    np_list, np_addr_list = shared_obj_list
    return np_list, np_addr_list
