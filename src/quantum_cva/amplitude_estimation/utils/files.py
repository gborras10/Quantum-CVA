'''
Functions related to file handling.
'''
import inspect
import os, sys
import pickle
import pytz
from datetime import datetime
from os import path
try:
    from google.colab import files
    using_colab = True
except:
     using_colab = False
     
def data_from_file(filename, silent = False):
    try:
        with open(filename, 'rb') as filehandle: 
            try:
                data = pickle.load(filehandle)
            except FileNotFoundError as e:
                print(f"> {e}. This may be fixable with scripts/fix_files.")
                sys.exit()
        if not silent:
            print(f"> File {filename} uploaded.")
    except FileNotFoundError:
        print(f"> File {filename} NOT uploaded: no such file. [data_from_file]")
        data = None
        raise
    return data

# Decorator.
def attempt_upload(f):
    def wrapper(*args, **kwargs):
        if len(kwargs) == 0:
            # Get f's default arguments (they're not passed to the decorator).
            bound_args = inspect.signature(f).bind(*args, **kwargs)
            bound_args.apply_defaults()
            bound_args = dict(bound_args.arguments)
            filename = bound_args["filename"]
        else:
            filename = kwargs["filename"]
            
        if filename is None:
            return f(*args, **kwargs)
        else:
            with open(filename, 'rb') as filehandle: 
                estdata = pickle.load(filehandle)
            print(f"> Uploaded data from file '{filename}'. "
                  f"[{f.__name__} @ attempt_upload]")
            # Assign attribute so the deccorated function can access 'estdata'.
            wrapper.estdata = estdata
            return f(*args, **kwargs)
    return wrapper

# Decorator.
def output_to_file(f):
    def wrapper(*args, **kwargs):
        output = f(*args, **kwargs)
        if output is None:
            return
        
        content, filename_stem = output
        filename = fix_filename(filename_stem)
        
        with open(filename, 'wb') as filehandle:
            pickle.dump(content, filehandle)
        if using_colab:
            files.download(filename)
            print(f"> File '{filename}' has been downloaded.")
        else:
            print(f"> File '{filename}' has been saved.")
        return output
    return wrapper

def fix_filename(filename, extension = None):
    '''
    Remove whitespaces and avoid overwriting files with the same name by 
    appending #N to the filename, N the smallest positive integer that avoids 
    collisions).
    
    extension: extension, if not included in the filename. 
    '''
    if extension is None:
        # Extension included in the filename; remove it, put back in the end.
        ext_start = filename.rfind('.')
        stem, extension = filename[:ext_start], filename[ext_start:]
    else:
        stem = filename
        
    # Remove whitespaces.
    stem.replace(" ", "")
    
    # Avoid writing over files with same name.
    append = 0
    while path.exists(stem + "#" + str(append) + extension):
        append += 1
        
    stem += "#" + str(append) + extension
    return stem

def keep_file(f, fix = True):
    if fix:
        filename = fix_filename(f.name)
        os.rename(f.name, filename)
    if using_colab:
        files.download(f.name)
        print(f"> File '{f.name}' has been downloaded.")
    else:
        print(f"> File '{f.name}' has been saved.")

@output_to_file
def save_as(data, filename):
    return data, filename

class PrintsToFile:
    def __init__(self, desc, silent = False):
        # File to be created later if using.
        self.f = None 
        timestamp = datetime.now(pytz.timezone('Portugal')).strftime("%d_%m_%Y_%H_%M")
        self.filename = fix_filename(f'{desc}_{timestamp}.txt')
        if not silent:
            print(f"\n> Will direct prints to file {self.filename}.")
        
    def __enter__(self):
        self.f = open(self.filename, 'a')
        self.original_stdout = sys.stdout
        sys.stdout = self.f

    def __exit__(self, *args):
        sys.stdout.close()
        sys.stdout = self.original_stdout
        self.f.close()

    def save_file(self):
        assert self.f is not None
        # Filename already fixed in init.
        keep_file(self.f, fix = False)