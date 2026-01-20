class DataNotFoundError(Exception):
    def __init__(self, value: str) -> None:
        super().__init__(value)


class UnknownWeatherModel(Exception):
    def __init__(self, value: str) -> None:
        super().__init__(value)


class NoDownloadMode(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UnsupportedGridDegreeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class UnknownModelCycle(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class HorizontalResolutionError(Exception):
    def __init__(self, *args):
        super().__init__(*args)
