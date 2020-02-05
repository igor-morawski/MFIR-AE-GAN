import HTPA32x32d
import os
import glob
import fnmatch
import re

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
        self._ids = ids

        self._update_files()
        
    def load(self):
        '''
        Loads in data triples.

        Returns
        -------
        #TODO
        '''
        self._update_files()
        translated_prefix = "{}".format(fnmatch.translate(self._prefix_pattern).strip("\\Z").strip("\\z"))
        prefixes_matched = set(re.findall(translated_prefix, file)[0] for file in self._files)
        def sample_triple(file_paths, prefix, ids):
            return [fnmatch.filter(file_paths, "*"+prefix+str(id)+"*") for id in ids]
        data = [sample_triple(self._files, prefix, self._ids) for prefix in prefixes_matched]
        return data 

    def _update_files(self):
        self._files = glob.glob(os.path.join(self._directory_path, "{}{}".format(self._pattern, self._file_extension)))


if __name__ == "__main__":
    dataset = Dataloader_RAM(ids = [121, 122, 123])
    data = dataset.load()