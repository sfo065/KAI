from parser.parser import parse
from pathlib import Path

if __name__ == "__main__":

    path = "flybildeprosjekt\AT\ATRapport_AgderØst2023-GSD07_FG-14583_V4_OrienteringsData\ATProsjekt"

    projection_info = parse(Path(path))

    print(projection_info["14583_01_001_20017"])
