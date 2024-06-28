from pathlib import Path

from parser.utils import get_files, read_and_parse_wkt

def parse(path: Path):

    files = get_files(path)

    wkt = files["prj"]

    crs = read_and_parse_wkt(wkt)

    return crs