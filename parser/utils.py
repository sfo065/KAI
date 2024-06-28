from pathlib import Path
import re


filetypes_of_interest = ["prj"]

def get_files(path: Path):

    files = {k: "" for k in filetypes_of_interest}

    for filetype in filetypes_of_interest:

        file = next(path.glob(f"*.{filetype}"), None)
        files[filetype] = file

    return files


def read_and_parse_wkt(prj_file: Path):
    """
    Parses the custom project file to extract photo information.
    
    :param prj_file: Path to the project file.
    :return: Dictionary containing photo data.
    """
    photos = {}
    in_photo_section = False
    line_counter = 0

    with open(prj_file, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith("$FOCAL_LENGTH"):
                photos["c"] = float(line.split(":")[1].strip())
            
            if line.startswith("$PRINCIPAL_POINT_PPA"):
                values = [float(line.split()[2]), float(line.split()[3])]
                photos["ppa_x"] = float(values[0])
                photos["ppa_y"] = float(values[1])

            if line == "$PHOTO":
                meta_data = {}
                in_photo_section = True
            
            elif line.startswith("$PHOTO_NUM") and in_photo_section:
                photo_num = line.split(":")[1].strip()
                photos[photo_num] = {}
            
            elif line.startswith("$EXT_ORI") and in_photo_section:
                line_counter += 1
            
            elif line_counter == 1 and in_photo_section:
                values = [float(x) for x in line.split()]
                # meta_data["c"] = values[0] we get this value from the camera definition
                meta_data["x"] = values[1]
                meta_data["y"] = values[2]
                meta_data["z"] = values[3]
                line_counter += 1
            
            elif line_counter == 2 and in_photo_section:
                values = [float(x) for x in line.split()]
                meta_data["e00"] = values[0]
                meta_data["e01"] = values[1]
                meta_data["e02"] = values[2]
                line_counter += 1
            
            elif line_counter == 3 and in_photo_section:
                values = [float(x) for x in line.split()]
                meta_data["e10"] = values[0]
                meta_data["e11"] = values[1]
                meta_data["e12"] = values[2]
                line_counter += 1
            
            elif line_counter == 4 and in_photo_section:
                values = [float(x) for x in line.split()]
                meta_data["e20"] = values[0]
                meta_data["e21"] = values[1]
                meta_data["e22"] = values[2]
                photos[photo_num] = meta_data
                line_counter = 0
            
            elif line == "$END" and in_photo_section:
                in_photo_section = False


    return photos