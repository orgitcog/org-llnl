import abc
import datetime
from typing import List
from apbuilder import exceptions


@staticmethod
def get_weather_model(weather_model: str):
    match weather_model.casefold():
        case "gfs":
            return GlobalForecastSystem()
        case "rap":
            return RapidRefresh()
        case "hrrr":
            return HighResolutionRapidRefresh()
        case "ifs":
            return IntegratedForecastSystem()
        case _:
            raise exceptions.UnknownWeatherModel(f"Unknown {weather_model=}")


class IWeatherModel(abc.ABC):
    @abc.abstractmethod
    def get_model_cycle(self, grid_degree: float) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_forecast_cycle(self, grid_degree: float) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_supported_grid_degrees(self) -> List[float]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_grid_degree_as_herbie_product(
        self, grid_degree: float, datetime: datetime.datetime
    ) -> str:
        raise NotImplementedError


class GlobalForecastSystem(IWeatherModel):
    def get_model_cycle(self, grid_degree: float = None) -> List[str]:
        # 4/day:00, 06, 12, 18UTC
        return [f"{hour:02}:00" for hour in range(0, 24, 6)]

    def get_forecast_cycle(self, grid_degree: float) -> List[str]:
        match grid_degree:
            case 0.25:
                return [f"{hour:02}:00" for hour in range(0, 24, 1)]
            case 0.5:
                return [f"{hour:02}:00" for hour in range(0, 24, 3)]
            case 1.0:
                return [f"{hour:02}:00" for hour in range(0, 24, 3)]
            case _:
                raise exceptions.UnsupportedGridDegreeError(
                    f"Grid Degree {grid_degree} is unsupported"
                )

    def get_supported_grid_degrees(self) -> List[float]:
        return [0.25, 0.5, 1.0]

    def get_grid_degree_as_herbie_product(
        self, grid_degree: float, date: datetime.datetime
    ) -> str:
        if date > datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc):
            match grid_degree:
                case 0.25:
                    return "pgrb2.0p25"
                case 0.50:
                    return "pgrb2.0p50"
                case 1.00:
                    return "pgrb2.1p00"
                case _:
                    raise exceptions.UnsupportedGridDegreeError(
                        f"Grid Degree {grid_degree} is unsupported"
                    )
        else:
            match grid_degree:
                case 0.50:
                    return "0.5-degree"
                case 1.00:
                    return "1.0-degree"
                case _:
                    raise exceptions.UnsupportedGridDegreeError(
                        f"Grid Degree {grid_degree} is unsupported"
                    )


class RapidRefresh(IWeatherModel):
    def get_model_cycle(self, grid_degree: float = None) -> List[str]:
        # 24/Day:00,01,02,… 23UTC
        return [f"{hour:02}:00" for hour in range(0, 24, 1)]

    def get_forecast_cycle(self, grid_degree: float) -> List[str]:
        match grid_degree:
            case 13:
                return [f"{hour:02}:00" for hour in range(0, 24, 1)]
            case _:
                raise exceptions.UnsupportedGridDegreeError(
                    f"Grid Degree {grid_degree} is unsupported"
                )

    def get_supported_grid_degrees(self) -> List[float]:
        return [13]

    def get_grid_degree_as_herbie_product(
        self, grid_degree: float, date: datetime.datetime = None
    ) -> str:
        match grid_degree:
            case 13:
                return "awp130pgrb"


class HighResolutionRapidRefresh(IWeatherModel):
    def get_model_cycle(self, grid_degree: float) -> List[str]:
        # 24/Day:00,01,02,… 23UTC
        return [f"{hour:02}:00" for hour in range(0, 24, 1)]

    def get_forecast_cycle(self, grid_degree: float) -> List[str]:
        match grid_degree:
            case 3:
                return [f"{hour:02}:00" for hour in range(0, 16, 1)]
            case _:
                raise exceptions.UnsupportedGridDegreeError(
                    f"Grid Degree {grid_degree} is unsupported"
                )

    def get_supported_grid_degrees(self) -> List[float]:
        return [3]

    def get_grid_degree_as_herbie_product(
        self, grid_degree: float, date: datetime.datetime = None
    ) -> str:
        match grid_degree:
            case 3:
                return "sfc"


class IntegratedForecastSystem(IWeatherModel):
    def get_model_cycle(self, grid_degree: float) -> List[str]:
        # 4/day:00, 06, 12, 18UTC
        return [f"{hour:02}:00" for hour in range(0, 24, 6)]

    def get_forecast_cycle(self, grid_degree: float) -> List[str]:
        return [f"{hour:02}:00" for hour in range(0, 24, 3)]

    def get_supported_grid_degrees(self) -> List[float]:
        return [0.25]

    def get_grid_degree_as_herbie_product(
        self, grid_degree: float, date: datetime.datetime = None
    ) -> str:
        match grid_degree:
            case 0.25:
                return "oper"
