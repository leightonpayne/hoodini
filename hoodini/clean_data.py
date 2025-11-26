from hoodini.utils.core import console
import shutil,os

class Cleaner:
    def __init__(self, obj):
        self.master = obj
        for key, val in vars(obj).items():
            setattr(self, key, val)

    def run(self):
        if self.keep is None:
            assembly_folder = os.path.join(self.output, "assembly_folder")
            if os.path.exists(assembly_folder):
                for root, dirs, files in os.walk(assembly_folder, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        os.remove(file_path)
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        os.rmdir(dir_path)
                shutil.rmtree(assembly_folder)
            else:
                console.print(f"Assembly folder does not exist: {assembly_folder}")
