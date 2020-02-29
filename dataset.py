try:
    import HTPA32x32d.tools
except ModuleNotFoundError:
    print("Run setup.sh to install HTPA32x32d module")
    raise ModuleNotFoundError
import os
import glob
import fnmatch
import re
import numpy as np

class Dataloader_RAM():
    '''
    
    Initializes a dataloader to load data into memory all at once. Do not use for large data sizes!
    This is a simplified dataloader for convinience only. Works only for files following the naming pattern:
    "YYYYMMDD_HHMM_ID{}.TXT" % int
    
    Parameters
    ----------
    directory_path : str, optional
        directory that contains all the *.TXT files, no directory structure is required, no recursive search for the files
    file_extension : str, optional
    prefix_pattern : str, optional
        Must be a glob pattern.
    ID_pattern : str, optional
        Must be a glob pattern.
    suffix : str, optional
        Regular string, not a pattern!


    This is a simplified dataloader that loads data of low-size directly into memory
    all at once!
    Do not use for large data sizes!
    '''
    def __init__(self, ids, directory_path = "data", file_extension = "TXT", prefix_pattern="{}_{}_ID".format("[0-9]"*8, "[0-9]"*4), ID_pattern = "*", suffix=""):
        self._directory_path = directory_path
        self._pattern = prefix_pattern + ID_pattern + suffix
        self._prefix_pattern= prefix_pattern
        self._ID_pattern = ID_pattern
        self._suffix = suffix
        self._file_extension = file_extension
        self.ids = ids

        self._update_files()
        
    def load(self):
        '''
        Loads in data triples.

        Returns
        -------
        list 
            [[arrays, timestamps]] both of lists shaped [sample][view]
        #TODO
        '''
        self._update_files()
        translated_prefix = "{}".format(fnmatch.translate(self._prefix_pattern).strip("\\Z").strip("\\z"))
        prefixes_matched = set(re.findall(translated_prefix, file)[0] for file in self._files)
        def sample_tuple(file_paths, prefix, ids):
            return [fnmatch.filter(file_paths, "*"+prefix+str(id)+"*")[0] for id in ids]
        file_names = [sample_tuple(self._files, prefix, self.ids) for prefix in prefixes_matched]
        def file_tuple2np(file_names_tuple, converter):
            sequences = []
            timestamps = [] 
            for file_name in file_names_tuple:
                data = converter(file_name)
                sequences.append(data[0])
                timestamps.append(data[1])
            return sequences, timestamps
        data = [[], []]
        for file_names_tuple in file_names:
            sequences, timestamps = file_tuple2np(file_names_tuple, HTPA32x32d.tools.txt2np) 
            data[0].append(sequences)
            data[1].append(timestamps)
        return data

    def _update_files(self):
        self._files = glob.glob(os.path.join(self._directory_path, "{}{}".format(self._pattern, self._file_extension)))

class Processor():
    def __init__(self):
        '''#TODO'''
        pass

    def align_timestamps(self, data):
        arrays, timestamps = data
        result = [[], []]
        for array_tuple, ts_tuple in zip(arrays, timestamps):
            indices = HTPA32x32d.tools.match_timesteps(*ts_tuple)
            result[0].append(HTPA32x32d.tools.resample_np_tuples(array_tuple, indices=indices))
            result[1].append(HTPA32x32d.tools.resample_timestamps(ts_tuple, indices=indices))
        return result

    def retime(self, data, step):
        '''#TODO'''
        arrays, timestamps = data
        result = [[], []]
        for array_tuple, ts_tuple in zip(arrays, timestamps):
            indices = HTPA32x32d.tools.match_timesteps(*ts_tuple)
            result[0].append(HTPA32x32d.tools.resample_np_tuples(array_tuple, step=step))
            result[1].append(HTPA32x32d.tools.resample_timestamps(ts_tuple, step=step))
        return result

class Data():
    def __init__(self, arrays, timestamps):
        ''' #TODO '''
if __name__ == "__main__":
    dataset = Dataloader_RAM(ids = [121, 122, 123])
    Processor = Processor()
    data_tuple = dataset.load()
    data = Processor.align_timestamps(data) # align frames ()
    data = Processor.retime(data, step = 3)