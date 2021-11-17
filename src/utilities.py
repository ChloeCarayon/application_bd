import zipfile

def unzip_file(old_path: str, new_path: str):
    """This function unzips a file.

    :param old_path: old path (for zip)
    :type old_path: str
    :param new_path: new path
    :type new_path: str
    """
    with zipfile.ZipFile(f"{old_path}.zip", "r") as zipref:
        zipref.extractall(new_path)